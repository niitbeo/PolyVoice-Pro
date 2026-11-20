use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{Emitter, Manager, RunEvent, WindowEvent};

const ENGINE_PORT: u16 = 18765;

struct EngineProcess(Mutex<Option<Child>>);
struct SavedOverlayBounds(Mutex<Option<(tauri::PhysicalPosition<i32>, tauri::PhysicalSize<u32>)>>);

fn engine_addr() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], ENGINE_PORT))
}

fn is_engine_running() -> bool {
    TcpStream::connect_timeout(&engine_addr(), Duration::from_millis(250)).is_ok()
}

fn open_engine_stderr() -> Stdio {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    if let Some(h) = home {
        let log_dir = h.join(".cache").join("voicerecore");
        if std::fs::create_dir_all(&log_dir).is_ok() {
            let log_path = log_dir.join("engine.log");
            if let Ok(file) = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(log_path)
            {
                return Stdio::from(file);
            }
        }
    }
    Stdio::null()
}

fn find_sidecar_binary(resource_dir: Option<&Path>) -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let bin_name = "voicerecore-engine.exe";
    #[cfg(not(target_os = "windows"))]
    let bin_name = "voicerecore-engine";

    let mut candidates = Vec::new();
    if let Some(res_dir) = resource_dir {
        candidates.push(
            res_dir
                .join("engine-bin")
                .join("voicerecore-engine")
                .join(bin_name),
        );
        candidates.push(
            res_dir
                .join("resources")
                .join("engine-bin")
                .join("voicerecore-engine")
                .join(bin_name),
        );
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(
                dir.join("engine-bin")
                    .join("voicerecore-engine")
                    .join(bin_name),
            );
            candidates.push(
                dir.join("resources")
                    .join("engine-bin")
                    .join("voicerecore-engine")
                    .join(bin_name),
            );
            candidates.push(
                dir.join("../Resources/engine-bin/voicerecore-engine")
                    .join(bin_name),
            );
            candidates.push(
                dir.join("../Resources/resources/engine-bin/voicerecore-engine")
                    .join(bin_name),
            );
            candidates.push(dir.join(bin_name));
        }
    }
    candidates.into_iter().find(|p| p.exists() && p.is_file())
}

fn find_engine_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("engine"));
        if let Some(parent) = cwd.parent() {
            candidates.push(parent.join("engine"));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("engine"));
            candidates.push(dir.join("../Resources/engine"));
        }
    }
    candidates
        .into_iter()
        .find(|p| p.join("server.py").exists())
}

fn spawn_local_engine(resource_dir: Option<&Path>) -> Option<Child> {
    if is_engine_running() {
        return None;
    }

    // 1. Packaged release (.exe / .msi / .dmg): prefer bundled standalone PyInstaller sidecar
    //    so end users need ZERO Python / uv / PyTorch installed!
    if let Some(sidecar) = find_sidecar_binary(resource_dir) {
        let mut cmd = Command::new(&sidecar);
        cmd.arg("--host")
            .arg("0.0.0.0")
            .arg("--port")
            .arg(ENGINE_PORT.to_string())
            .stdout(Stdio::null())
            .stderr(open_engine_stderr());
        if let Some(parent) = sidecar.parent() {
            cmd.current_dir(parent);
        }
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        if let Ok(child) = cmd.spawn() {
            return Some(child);
        }
    }

    // 2. Development mode (`pnpm tauri dev`): use managed .venv python directly (<200ms cold start)
    let engine_dir = find_engine_dir()?;
    let server_py = engine_dir.join("server.py");

    #[cfg(target_os = "windows")]
    let venv_python = engine_dir.join(".venv").join("Scripts").join("python.exe");
    #[cfg(not(target_os = "windows"))]
    let venv_python = engine_dir.join(".venv").join("bin").join("python");

    if venv_python.exists() {
        let mut cmd = Command::new(&venv_python);
        cmd.arg(&server_py)
            .arg("--host")
            .arg("0.0.0.0")
            .arg("--port")
            .arg(ENGINE_PORT.to_string())
            .current_dir(&engine_dir)
            .stdout(Stdio::null())
            .stderr(open_engine_stderr());
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        if let Ok(child) = cmd.spawn() {
            return Some(child);
        }
    }

    // 3. Fallback to `uv run`
    let mut cmd = Command::new("uv");
    cmd.arg("run")
        .arg("--project")
        .arg(&engine_dir)
        .arg("python")
        .arg(&server_py)
        .arg("--host")
        .arg("0.0.0.0")
        .arg("--port")
        .arg(ENGINE_PORT.to_string())
        .stdout(Stdio::null())
        .stderr(open_engine_stderr());
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd.spawn().ok()
}

fn request_http_shutdown() -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(&engine_addr(), Duration::from_millis(500)) else {
        return false;
    };
    let _ = stream.set_write_timeout(Some(Duration::from_millis(1500)));
    let _ = stream.set_read_timeout(Some(Duration::from_millis(1500)));
    let req = format!(
        "POST /api/shutdown HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        ENGINE_PORT
    );
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let _ = stream.flush();
    let mut buf = [0u8; 256];
    let _ = stream.read(&mut buf);
    true
}

fn wait_child_exit(child: &mut Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(_)) => return true,
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => return true,
        }
    }
    matches!(child.try_wait(), Ok(Some(_)))
}

fn force_kill_tree(child: &mut Child) {
    let pid = child.id();

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        let mut tk = Command::new("taskkill");
        tk.args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .creation_flags(0x08000000); // CREATE_NO_WINDOW
        let _ = tk.status();
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if wait_child_exit(child, Duration::from_millis(400)) {
            return;
        }
    }

    let _ = child.kill();
    let _ = child.wait();
}

fn stop_engine_internal(state: &EngineProcess) {
    let owned_child = state.0.lock().ok().and_then(|mut g| g.take());
    let engine_up = is_engine_running();

    if owned_child.is_some() || engine_up {
        let _ = request_http_shutdown();
    }

    if let Some(mut proc_child) = owned_child {
        if !wait_child_exit(&mut proc_child, Duration::from_millis(2000)) {
            force_kill_tree(&mut proc_child);
        }
    } else if engine_up {
        // Wait briefly for an externally launched engine to release port & file locks
        let deadline = Instant::now() + Duration::from_millis(1500);
        while Instant::now() < deadline && is_engine_running() {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

#[tauri::command]
fn get_engine_url() -> String {
    format!("http://127.0.0.1:{}", ENGINE_PORT)
}

#[tauri::command]
fn stop_local_engine(state: tauri::State<'_, EngineProcess>) -> Result<(), String> {
    stop_engine_internal(&state);
    Ok(())
}

#[tauri::command]
fn toggle_overlay(app: tauri::AppHandle) -> Result<bool, String> {
    let win = app
        .get_webview_window("overlay")
        .ok_or_else(|| "Overlay window not found".to_string())?;
    let visible = win.is_visible().map_err(|e| e.to_string())?;
    if visible {
        win.hide().map_err(|e| e.to_string())?;
        let _ = app.emit("overlay-visibility", false);
        Ok(false)
    } else {
        win.show().map_err(|e| e.to_string())?;
        let _ = app.emit("overlay-visibility", true);
        Ok(true)
    }
}

#[tauri::command]
fn set_overlay_clickthrough(app: tauri::AppHandle, ignore: bool) -> Result<(), String> {
    let win = app
        .get_webview_window("overlay")
        .ok_or_else(|| "Overlay window not found".to_string())?;
    win.set_ignore_cursor_events(ignore)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn set_overlay_always_on_top(app: tauri::AppHandle, on_top: bool) -> Result<bool, String> {
    let win = app
        .get_webview_window("overlay")
        .ok_or_else(|| "Overlay window not found".to_string())?;
    win.set_always_on_top(on_top).map_err(|e| e.to_string())?;
    Ok(on_top)
}

#[tauri::command]
fn toggle_overlay_fill_monitor(
    app: tauri::AppHandle,
    saved: tauri::State<'_, SavedOverlayBounds>,
) -> Result<bool, String> {
    let win = app
        .get_webview_window("overlay")
        .ok_or_else(|| "Overlay window not found".to_string())?;

    let mut guard = saved.0.lock().map_err(|e| e.to_string())?;
    if let Some((pos, size)) = guard.take() {
        win.set_position(tauri::Position::Physical(pos))
            .map_err(|e| e.to_string())?;
        win.set_size(tauri::Size::Physical(size))
            .map_err(|e| e.to_string())?;
        return Ok(false);
    }

    if let (Ok(cur_pos), Ok(cur_size)) = (win.outer_position(), win.outer_size()) {
        *guard = Some((cur_pos, cur_size));
    }

    if let Ok(Some(mon)) = win.current_monitor() {
        let mpos = *mon.position();
        let msize = *mon.size();
        win.set_position(tauri::Position::Physical(mpos))
            .map_err(|e| e.to_string())?;
        win.set_size(tauri::Size::Physical(msize))
            .map_err(|e| e.to_string())?;
        return Ok(true);
    }

    win.maximize().map_err(|e| e.to_string())?;
    Ok(true)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .manage(EngineProcess(Mutex::new(None)))
        .manage(SavedOverlayBounds(Mutex::new(None)))
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state == tauri_plugin_global_shortcut::ShortcutState::Pressed {
                        let _ = app.emit("toggle-listening", ());
                    }
                })
                .build(),
        )
        .setup(|app| {
            let res_dir = app.path().resource_dir().ok();
            let child = spawn_local_engine(res_dir.as_deref());
            if let Some(state) = app.try_state::<EngineProcess>() {
                if let Ok(mut guard) = state.0.lock() {
                    *guard = child;
                }
            }

            use tauri_plugin_global_shortcut::GlobalShortcutExt;
            let _ = app.global_shortcut().register("CommandOrControl+Shift+Space");
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == "main" {
                if let WindowEvent::CloseRequested { .. } = event {
                    let app_handle = window.app_handle();
                    if let Some(overlay) = app_handle.get_webview_window("overlay") {
                        let _ = overlay.destroy();
                    }
                    if let Some(state) = app_handle.try_state::<EngineProcess>() {
                        stop_engine_internal(&state);
                    }
                    app_handle.exit(0);
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_engine_url,
            stop_local_engine,
            toggle_overlay,
            set_overlay_clickthrough,
            set_overlay_always_on_top,
            toggle_overlay_fill_monitor
        ])
        .build(tauri::generate_context!())
        .expect("error while building VOICERECORE");

    app.run(|app_handle, event| {
        if let RunEvent::Exit = event {
            if let Some(state) = app_handle.try_state::<EngineProcess>() {
                stop_engine_internal(&state);
            }
        }
    });
}
