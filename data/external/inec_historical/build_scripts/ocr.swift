// usage: swift ocr.swift <image> > out.tsv
// prints: x\ty\tw\th\tconfidence\ttext  (normalized coords, origin top-left)
import Foundation
import Vision
import AppKit

let path = CommandLine.arguments[1]
guard let img = NSImage(contentsOfFile: path),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    FileHandle.standardError.write("cannot load image\n".data(using: .utf8)!)
    exit(1)
}
let req = VNRecognizeTextRequest()
req.recognitionLevel = .accurate
req.usesLanguageCorrection = false
req.minimumTextHeight = 0.004
let handler = VNImageRequestHandler(cgImage: cg, options: [:])
try handler.perform([req])
for obs in (req.results ?? []) {
    guard let c = obs.topCandidates(1).first else { continue }
    let b = obs.boundingBox
    let line = String(format: "%.5f\t%.5f\t%.5f\t%.5f\t%.3f\t", b.minX, 1.0 - b.maxY, b.width, b.height, c.confidence) + c.string
    print(line)
}
