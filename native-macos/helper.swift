import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit
import Speech

// MARK: - JSON Stdout Helpers

let stdoutLock = NSLock()

func emitJSON(_ dict: [String: Any]) {
    stdoutLock.lock()
    defer { stdoutLock.unlock() }
    if let data = try? JSONSerialization.data(withJSONObject: dict, options: []),
       let line = String(data: data, encoding: .utf8) {
        FileHandle.standardOutput.write((line + "\n").data(using: .utf8)!)
    }
}

func emitError(_ message: String) {
    emitJSON(["type": "error", "message": message])
}

// MARK: - PCM Resampler & Converter (Any Float32/Int16 CMSampleBuffer or AVAudioPCMBuffer -> 16kHz Mono Int16)

final class PCM16kConverter {
    private var converter: AVAudioConverter?
    private var lastInputFormat: AVAudioFormat?
    let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: 16000,
        channels: 1,
        interleaved: true
    )!

    func convert(buffer: AVAudioPCMBuffer) -> (data: Data, rms: Float)? {
        let inFmt = buffer.format
        if converter == nil || lastInputFormat != inFmt {
            converter = AVAudioConverter(from: inFmt, to: targetFormat)
            lastInputFormat = inFmt
        }
        guard let conv = converter else { return nil }

        let ratio = targetFormat.sampleRate / inFmt.sampleRate
        let outCap = AVAudioFrameCount(max(1, Int(Double(buffer.frameLength) * ratio) + 64))
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCap) else {
            return nil
        }

        var supplied = false
        var err: NSError?
        let status = conv.convert(to: outBuf, error: &err) { _, outStatus in
            if supplied {
                outStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            outStatus.pointee = .haveData
            return buffer
        }
        guard status != .error, err == nil, outBuf.frameLength > 0,
              let int16Ptr = outBuf.int16ChannelData?[0] else {
            return nil
        }

        let count = Int(outBuf.frameLength)
        var sumSq: Double = 0
        for i in 0..<count {
            let v = Double(int16Ptr[i]) / 32768.0
            sumSq += v * v
        }
        let rms = Float(sqrt(sumSq / Double(max(count, 1))))
        let data = Data(bytes: int16Ptr, count: count * MemoryLayout<Int16>.size)
        return (data, rms)
    }

    func convertSampleBuffer(_ sampleBuffer: CMSampleBuffer) -> (pcmBuffer: AVAudioPCMBuffer, data: Data, rms: Float)? {
        guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbdPtr = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc) else {
            return nil
        }
        var asbd = asbdPtr.pointee
        guard let format = AVAudioFormat(streamDescription: &asbd) else { return nil }
        let frameCount = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard frameCount > 0,
              let pcmBuf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return nil
        }
        pcmBuf.frameLength = frameCount
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(frameCount),
            into: pcmBuf.mutableAudioBufferList
        )
        guard status == noErr else { return nil }
        guard let conv = convert(buffer: pcmBuf) else { return nil }
        return (pcmBuf, conv.data, conv.rms)
    }
}

// MARK: - ScreenCaptureKit System Audio Streamer (macOS 13+)

final class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let queue = DispatchQueue(label: "voicerecore.sck.audio")
    private let converter = PCM16kConverter()
    private let onBuffer: (AVAudioPCMBuffer, Data, Float) -> Void
    private let onError: ((Error) -> Void)?

    init(
        onBuffer: @escaping (AVAudioPCMBuffer, Data, Float) -> Void,
        onError: ((Error) -> Void)? = nil
    ) {
        self.onBuffer = onBuffer
        self.onError = onError
    }

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            throw NSError(domain: "VOICERECORE", code: 1, userInfo: [NSLocalizedDescriptionKey: "Không tìm thấy màn hình để thu âm thanh hệ thống."])
        }

        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 16000
        config.channelCount = 1
        // Minimal dummy video settings so macOS ScreenCaptureKit uses virtually zero GPU/CPU
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let scStream = SCStream(filter: filter, configuration: config, delegate: self)
        try scStream.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        try await scStream.startCapture()
        self.stream = scStream
    }

    func stop() async {
        if let s = stream {
            try? await s.stopCapture()
            stream = nil
        }
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }
        if let res = converter.convertSampleBuffer(sampleBuffer) {
            onBuffer(res.pcmBuffer, res.data, res.rms)
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        if let onError = onError {
            onError(error)
        } else {
            emitError("Luồng thu âm hệ thống dừng: \(error.localizedDescription)")
        }
    }
}

// MARK: - Apple Native SFSpeechRecognizer Continuous Engine (Neural Engine)

final class AppleSpeechRunner {
    private let localeId: String
    private let source: String // "mic" | "system"
    private let recognizer: SFSpeechRecognizer
    private let audioEngine = AVAudioEngine()
    private var sysCapture: SystemAudioCapture?
    private let pcmConv = PCM16kConverter()

    private let stateLock = NSLock()
    private var currentRequest: SFSpeechAudioBufferRecognitionRequest?
    private var currentTask: SFSpeechRecognitionTask?
    private var taskStartTime = Date()
    private var lastText = ""
    private var lastTextChangeTime = Date()
    private var lastLevelEmit = Date.distantPast
    private var running = false
    private var rotating = false
    private var watchdogTimer: DispatchSourceTimer?

    init(localeId: String, source: String) throws {
        self.localeId = localeId
        self.source = source
        guard let rec = SFSpeechRecognizer(locale: Locale(identifier: localeId)) else {
            throw NSError(domain: "VOICERECORE", code: 2, userInfo: [NSLocalizedDescriptionKey: "macOS không hỗ trợ ngôn ngữ \(localeId)"])
        }
        self.recognizer = rec
    }

    private func requestAuth() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { status in
                cont.resume(returning: status)
            }
        }
    }

    func run() async {
        // Strict 100% Offline guarantee: check on-device support BEFORE touching TCC or network!
        guard recognizer.supportsOnDeviceRecognition else {
            emitError("Ngôn ngữ \(localeId) chưa tải gói nhận dạng Offline (On-Device) trên macOS — từ chối gửi âm thanh lên máy chủ Apple để bảo đảm 100% Offline.")
            exit(1)
        }

        let auth = await requestAuth()
        guard auth == .authorized else {
            emitError("Chưa có quyền Nhận dạng giọng nói macOS (Cài đặt → Quyền riêng tư & Bảo mật → Nhận dạng giọng nói). Status=\(auth.rawValue)")
            exit(1)
        }

        guard recognizer.isAvailable else {
            emitError("Apple SpeechRecognizer chưa sẵn sàng cho \(localeId).")
            exit(1)
        }

        running = true
        startNewRecognitionTask()
        startWatchdog()

        emitJSON([
            "type": "ready",
            "locale": localeId,
            "on_device": recognizer.supportsOnDeviceRecognition,
            "source": source,
        ])

        do {
            if source == "system" {
                let cap = SystemAudioCapture { [weak self] pcmBuf, _, rms in
                    self?.feedAudio(pcmBuf, rms: rms)
                }
                self.sysCapture = cap
                try await cap.start()
            } else {
                try startMicCapture()
            }
        } catch {
            emitError("Không thể mở nguồn âm thanh (\(source)): \(error.localizedDescription)")
            exit(1)
        }

        // Keep process alive until terminated via SIGTERM/SIGINT
        while running {
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
    }

    private func startMicCapture() throws {
        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 2048, format: format) { [weak self] buffer, _ in
            guard let self = self else { return }
            let rms = self.pcmConv.convert(buffer: buffer)?.rms ?? 0.0
            self.feedAudio(buffer, rms: rms)
        }
        audioEngine.prepare()
        try audioEngine.start()
    }

    private func feedAudio(_ buffer: AVAudioPCMBuffer, rms: Float) {
        stateLock.lock()
        let req = currentRequest
        let now = Date()
        let shouldEmitLevel = now.timeIntervalSince(lastLevelEmit) >= 0.08
        if shouldEmitLevel { lastLevelEmit = now }
        stateLock.unlock()

        req?.append(buffer)

        if shouldEmitLevel {
            let level = min(1.0, Double(rms) * 6.5)
            emitJSON(["type": "level", "level": (level * 1000).rounded() / 1000])
        }
    }

    private func startNewRecognitionTask() {
        stateLock.lock()
        currentTask?.cancel()
        currentTask = nil
        currentRequest?.endAudio()
        currentRequest = nil

        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        if #available(macOS 13.0, *) {
            req.addsPunctuation = true
        }
        req.requiresOnDeviceRecognition = true
        currentRequest = req
        taskStartTime = Date()
        lastText = ""
        lastTextChangeTime = Date()
        rotating = false
        stateLock.unlock()

        let task = recognizer.recognitionTask(with: req) { [weak self] result, error in
            guard let self = self else { return }
            if let result = result {
                let text = result.bestTranscription.formattedString.trimmingCharacters(in: .whitespacesAndNewlines)
                let now = Date()

                self.stateLock.lock()
                let changed = !text.isEmpty && text != self.lastText
                if changed {
                    self.lastText = text
                    self.lastTextChangeTime = now
                }
                let wordCount = text.split(separator: " ").count
                let age = now.timeIntervalSince(self.taskStartTime)
                self.stateLock.unlock()

                if result.isFinal {
                    if !text.isEmpty {
                        emitJSON(["type": "final", "text": text, "asr_ms": 25.0])
                    }
                    self.rotateIfNeeded(emitPending: false)
                } else if changed {
                    emitJSON(["type": "partial", "text": text, "asr_ms": 20.0])
                    // Auto-commit long continuous speech (e.g. YouTube) cleanly as fuller paragraphs
                    if (wordCount >= 32 && (text.hasSuffix(".") || text.hasSuffix("?") || text.hasSuffix("!")))
                        || wordCount >= 44
                        || age >= 45.0 {
                        self.rotateIfNeeded(emitPending: true)
                    }
                }
            } else if let error = error {
                let nsErr = error as NSError
                // Ignore expected cancellation codes (216 / 203 / 301) during seamless rotation
                if nsErr.code != 216 && nsErr.code != 203 && nsErr.code != 301 {
                    self.rotateIfNeeded(emitPending: true)
                }
            }
        }

        stateLock.lock()
        currentTask = task
        stateLock.unlock()
    }

    private func rotateIfNeeded(emitPending: Bool) {
        stateLock.lock()
        if !running || rotating {
            stateLock.unlock()
            return
        }
        rotating = true
        let pending = lastText
        lastText = ""
        let oldReq = currentRequest
        let oldTask = currentTask
        currentRequest = nil
        currentTask = nil
        stateLock.unlock()

        if emitPending && !pending.isEmpty {
            emitJSON(["type": "final", "text": pending, "asr_ms": 25.0])
        }

        oldReq?.endAudio()
        oldTask?.cancel()

        DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + 0.04) { [weak self] in
            self?.startNewRecognitionTask()
        }
    }

    private func startWatchdog() {
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        timer.schedule(deadline: .now() + 0.3, repeating: 0.25)
        timer.setEventHandler { [weak self] in
            guard let self = self else { return }
            self.stateLock.lock()
            let txt = self.lastText
            let idle = Date().timeIntervalSince(self.lastTextChangeTime)
            let age = Date().timeIntervalSince(self.taskStartTime)
            let isRot = self.rotating
            self.stateLock.unlock()

            if isRot { return }
            // Commit paragraph automatically after 1.6s genuine pause, or rotate before Apple's 60s cap
            if (!txt.isEmpty && idle >= 1.6) || age >= 46.0 {
                self.rotateIfNeeded(emitPending: !txt.isEmpty)
            }
        }
        timer.resume()
        watchdogTimer = timer
    }
}

// MARK: - CLI Entry Point

@main
struct MacHelperMain {
    static func main() async {
        let args = CommandLine.arguments
        func argValue(_ key: String, default def: String) -> String {
            if let idx = args.firstIndex(of: key), idx + 1 < args.count {
                return args[idx + 1]
            }
            return def
        }

        let mode = argValue("--mode", default: "check")

        if mode == "check" {
            let sfLocales = SFSpeechRecognizer.supportedLocales()
            var onDevice: [String] = []
            var available: [String] = []
            for loc in sfLocales {
                let norm = loc.identifier.replacingOccurrences(of: "_", with: "-")
                if let r = SFSpeechRecognizer(locale: loc), r.isAvailable {
                    available.append(norm)
                    if r.supportsOnDeviceRecognition {
                        onDevice.append(norm)
                    }
                }
            }
            emitJSON([
                "type": "capabilities",
                "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "on_device_locales": onDevice.sorted(),
                "available_locales": available.sorted(),
                "system_audio": true,
            ])
            return
        }

        if mode == "system-pcm" {
            // Stream raw 16kHz mono int16 PCM from ScreenCaptureKit directly to stdout!
            // IMPORTANT: Never write JSON/text to stdout in raw PCM mode; send errors to stderr and exit.
            let capture = SystemAudioCapture(
                onBuffer: { _, pcmData, _ in
                    FileHandle.standardOutput.write(pcmData)
                },
                onError: { err in
                    fputs("ERROR: Luồng thu âm hệ thống dừng: \(err.localizedDescription)\n", stderr)
                    fflush(stderr)
                    exit(1)
                }
            )
            do {
                try await capture.start()
                while true {
                    try? await Task.sleep(nanoseconds: 500_000_000)
                }
            } catch {
                fputs("ERROR: \(error.localizedDescription)\n", stderr)
                fflush(stderr)
                exit(1)
            }
        }

        if mode == "apple-asr" {
            let locale = argValue("--locale", default: "vi-VN")
            let source = argValue("--source", default: "mic")
            do {
                let runner = try AppleSpeechRunner(localeId: locale, source: source)
                await runner.run()
            } catch {
                emitError(error.localizedDescription)
                exit(1)
            }
        }

        emitError("Unknown mode: \(mode)")
        exit(2)
    }
}
