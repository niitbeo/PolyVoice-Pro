use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{Emitter, Manager, RunEvent};

const ENGINE_PORT: u16 = 18765;

struct EngineProcess(Mutex<Option<Child>>);
struct SavedOverlayBounds(Mutex<Option<(tauri::PhysicalPosition<i32>, tauri::PhysicalSize<u32>)>>);

fn is_engine_running() -> bool {
    TcpStream::connect(("127.0.0.1", ENGINE_PORT)).is_ok()
}

fn find_sidecar_binary() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let bin_name = "voicerecore-engine.exe";
    #[cfg(not(target_os = "windows"))]
    let bin_name = "voicerecore-engine";

    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("engine-bin").join("voicerecore-engine").join(bin_name));
            candidates.push(
                dir.join("resources")
                    .join("engine-bin")
                    .join("voicerecore-engine")
                    .join(bin_name),
            );
            candidates.push(dir.join("../Resources/engine-bin/voicerecore-engine").join(bin_name));
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

fn spawn_local_engine() -> Option<Child> {
    if is_engine_running() {
        return None;
    }

    // 1. Packaged release (.exe / .msi / .dmg): prefer bundled standalone PyInstaller sidecar
    //    so end users need ZERO Python / uv / PyTorch installed!
    if let Some(sidecar) = find_sidecar_binary() {
        let mut cmd = Command::new(&sidecar);
        cmd.arg("--host")
            .arg("0.0.0.0")
            .arg("--port")
            .arg(ENGINE_PORT.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
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
            .stderr(Stdio::null());
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
        .stderr(Stdio::null());
    cmd.spawn().ok()
}

#[tauri::command]
fn get_engine_url() -> String {
    format!("http://127.0.0.1:{}", ENGINE_PORT)
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
        let _ = win.set_always_on_top(true);
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
    let child = spawn_local_engine();

    let app = tauri::Builder::default()
        .manage(EngineProcess(Mutex::new(child)))
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
            use tauri_plugin_global_shortcut::GlobalShortcutExt;
            let _ = app.global_shortcut().register("CommandOrControl+Shift+Space");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_engine_url,
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
                if let Ok(mut guard) = state.0.lock() {
                    if let Some(mut proc_child) = guard.take() {
                        let _ = proc_child.kill();
                    }
                }
            }
        }
    });
}
