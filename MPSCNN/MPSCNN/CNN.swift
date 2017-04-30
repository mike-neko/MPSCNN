//
//  CNN.swift
//  MPSCNN
//
//  Created by M.Ike on 2017/04/30.
//  Copyright © 2017年 M.Ike. All rights reserved.
//

import Foundation
import MetalKit
import MetalPerformanceShaders

// TODO: リファクタ
extension CNN {
    static var weightParamPrefix = "weights_"
    static var biasParamPrefix = "bias_"
    static var paramExtension = ".dat"
    
    typealias Format = (w: Int, h: Int, ch: Int)
    typealias TempFormat = (w: Int, h: Int, ch: Int, count: Int)
    
    // memo: 色空間は呼び出し側で合わせる
    enum Input {
        // TODO: 未実装
        case image(UIImage)
        //        case imageName(String)
        case texture(MTLTexture)
        //        case path(URL)
        
        fileprivate func toTexture(device: MTLDevice) -> MTLTexture? {
            switch self {
            case let .image(image):
                guard let cgImage = image.cgImage else { return nil }
                let textureLoader = MTKTextureLoader(device: device)
                return try? textureLoader.newTexture(with: cgImage, options: nil)
            case let .texture(tex): return tex
            }
        }
    }
}

class CNN {
    private let commandQueue: MTLCommandQueue
    
    private var kernelList = [(Layer, MPSCNNKernel)]()
    private var tempList = [String: TempFormat]()
    private var resultImage: MPSImage!
    private var inputFormat: MPSImageDescriptor!
    
    private var isReady = false
    
    private let lanczos: MPSImageLanczosScale
    private let scale: MPSCNNNeuronLinear
    
    required init?() {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        guard MPSSupportsMTLDevice(device) else { return nil }
        
        commandQueue = device.makeCommandQueue()
        lanczos = MPSImageLanczosScale(device: device)
        scale = MPSCNNNeuronLinear(device: device, a: 2, b: -1)
    }
    
    func setup(model: CNNModel) -> Bool {
        isReady = false
        // Layerとimageの生成
        let device = commandQueue.device
        kernelList = []
        tempList = [:]
        var images = [String: (Format, Int)]()
        var layers = [Layer]()
        
        for item in model.makeModel().enumerated() {
            // image
            var layer = item.element
            
            if layer.outputImage == "" {
                layer.outputImage = layer.name
            }
            
            var inputFormat = (Format(w: 0, h: 0, ch: 0), 0)
            if item.offset == 0 {
                // first
                inputFormat = (model.inputFormat, 0)
            } else {
                if layer.inputImage == "" {
                    layer.inputImage = layers[item.offset - 1].name
                }
                guard let fmt = images[layer.inputImage] else { return false }
                inputFormat = fmt
                inputFormat.1 += 1
            }
            
            var outputFormat = layer.type.outputSize(input: inputFormat.0)
            if let exist = images[layer.outputImage] {
                guard exist.0.w == inputFormat.0.w && exist.0.h == inputFormat.0.h else { return false }
                layer.outputOffset = inputFormat.0.ch
                outputFormat.ch += exist.0.ch
            }
            images[layer.outputImage] = (outputFormat, 0)
            
            layers += [layer]
        }
        
        //        print("input:\n => \(model.inputFormat)")
        //        for layer in layers {
        //            print(layer.summary)
        //            let inImage = images[layer.inputImage] ?? model.inputFormat
        //            print("\(inImage) => \(layer.outputImage)\(images[layer.outputImage]!)")
        //        }
        //        images.forEach { print("\($0.key): \($0.value)") }
        
        for layer in layers {
            guard let format = images[layer.inputImage],
                let kernel = layer.make(device: device, inputFormat: format.0) else {
                    return false
            }
            kernelList += [(layer, kernel)]
        }
        
        guard let last = kernelList.last?.0.name else { return false }
        guard let result = images.removeValue(forKey: last)?.0 else { return false }
        images.forEach {
            let fmt = $0.value.0
            tempList[$0.key] = (w: fmt.w, h: fmt.h, ch: fmt.ch, count: $0.value.1)
        }
        
        inputFormat = MPSImageDescriptor(channelFormat: .float16,
                                         width: model.inputFormat.w,
                                         height: model.inputFormat.h,
                                         featureChannels: model.inputFormat.ch)
        resultImage = MPSImage(device: device,
                               imageDescriptor: MPSImageDescriptor(channelFormat: .float16,
                                                                   width: result.w, height: result.h,
                                                                   featureChannels: result.ch))
        isReady = true
        return true
    }
    
    func run(input: Input) -> [Float] {
        guard isReady else { return [] }
        guard let inputDesc = inputFormat else { return [] }
        
        guard let sourceTexture = input.toTexture(device: commandQueue.device) else {
            return []
        }
        
        autoreleasepool {
            let commandBuffer = commandQueue.makeCommandBuffer()
            
            var images = [String : MPSImage]()
            let descs: [MPSImageDescriptor] = tempList.map {
                let desc = MPSImageDescriptor(channelFormat: .float16,
                                              width: $0.value.w, height: $0.value.h,
                                              featureChannels: $0.value.ch)
                let temp = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: desc)
                temp.readCount = $0.value.count
                images[$0.key] = temp
                return desc
                } + [inputDesc, inputDesc]
            
            let srcImage = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: inputDesc)
            let preImage = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: inputDesc)
            
            images[""] = srcImage
            images[kernelList[kernelList.count - 1].0.name] = resultImage
            
            MPSTemporaryImage.prefetchStorage(with: commandBuffer, imageDescriptorList: descs)
            
            lanczos.encode(commandBuffer: commandBuffer, sourceTexture: sourceTexture, destinationTexture: preImage.texture)
            scale.encode(commandBuffer: commandBuffer, sourceImage: preImage, destinationImage: srcImage)
            kernelList.forEach {
                guard let inImage = images[$0.0.inputImage],
                    let outImage = images[$0.0.outputImage] else { return }
                $0.1.encode(commandBuffer: commandBuffer, sourceImage: inImage, destinationImage: outImage)
            }
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        return resultImage.float16ToFloatArray()
    }
}

