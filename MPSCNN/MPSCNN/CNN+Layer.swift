//
//  CNN+Layer.swift
//  MPSCNN
//
//  Created by M.Ike on 2017/04/30.
//  Copyright © 2017年 M.Ike. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

extension CNN {
    struct Layer {
        var name: String
        var type: SpecificData
        var inputImage: String
        var outputImage: String
        var outputOffset: Int          // TODO: 未実装
        
        private func loadParameter<T>(body: (UnsafePointer<Float>, UnsafePointer<Float>) -> T) -> T? {
            guard let weightPath = Bundle.main.url(forResource: CNN.weightParamPrefix + name, withExtension: CNN.paramExtension),
                let rawWeight = try? Data(contentsOf: weightPath) else {
                    return nil
            }
            guard let biasPath = Bundle.main.url(forResource: CNN.biasParamPrefix + name, withExtension: CNN.paramExtension),
                let rawBias = try? Data(contentsOf: biasPath) else {
                    return nil
            }
            
            return rawWeight.withUnsafeBytes { (weight: UnsafePointer<Float>) -> T in
                return rawBias.withUnsafeBytes { (bias: UnsafePointer<Float>) -> T in
                    return body(weight, bias)
                }
            }
        }
        
        func make(device: MTLDevice, inputFormat: Format) -> MPSCNNKernel? {
            switch type {
            case let .convolution(ch, w, h, stride, _, activation):
                let desc = MPSCNNConvolutionDescriptor(kernelWidth: w, kernelHeight: h,
                                                       inputFeatureChannels: inputFormat.ch,
                                                       outputFeatureChannels: ch,
                                                       neuronFilter: activation.makeNeuron(device: device))
                desc.strideInPixelsX = stride.0
                desc.strideInPixelsY = stride.1
                
                return loadParameter(body: { (weights, bias) -> MPSCNNKernel in
                    let conv = MPSCNNConvolution(device: device, convolutionDescriptor: desc,
                                                 kernelWeights: weights, biasTerms: bias, flags: .none)
                    conv.offset = MPSOffset(x: Int(w) / 2, y: Int(h) / 2, z: 0)
                    return conv
                })
            case let .fullyConnected(ch, activation):
                let desc = MPSCNNConvolutionDescriptor(kernelWidth: inputFormat.w, kernelHeight: inputFormat.h,
                                                       inputFeatureChannels: inputFormat.ch,
                                                       outputFeatureChannels: ch,
                                                       neuronFilter: activation.makeNeuron(device: device))
                
                return loadParameter(body: { (weights, bias) -> MPSCNNKernel in
                    return MPSCNNFullyConnected(device: device, convolutionDescriptor: desc,
                                                kernelWeights: weights, biasTerms: bias, flags: .none)
                })
                
            case let .maxPooling(size, stride):
                let pool = MPSCNNPoolingMax(device: device, kernelWidth: size.0, kernelHeight: size.1,
                                            strideInPixelsX: stride.0, strideInPixelsY: stride.1)
                pool.offset = MPSOffset(x: size.0 / 2, y: size.1 / 2, z: 0)
                pool.edgeMode = .clamp  // TODO: !
                return pool
            case .softmax:
                return MPSCNNSoftMax(device: device)
            }
        }
        
        var summary: String {
            return "\(name)(\(type.name)): \(type.summary)"
        }
    }
}

extension CNN.Layer {
    enum SpecificData {
        case convolution(ch: Int, w: Int, h: Int, stride: (Int, Int), padding: Padding, activation: Activation)
        case fullyConnected(ch: Int, activation: Activation)
        
        case maxPooling(size: (Int, Int), stride: (Int, Int))
        
        case softmax
        
        func outputSize(input: CNN.Format) -> CNN.Format {
            switch self {
            case let .convolution(ch, w, h, stride, padding, _):
                switch padding {
                case .same:
                    return (w: Int(ceil(Float(input.w) / Float(stride.0))),
                            h: Int(ceil(Float(input.h) / Float(stride.1))),
                            ch: ch)
                case .valid:
                    return (w: Int(ceil(Float(input.w - w + 1) / Float(stride.0))),
                            h: Int(ceil(Float(input.h - h + 1) / Float(stride.1))),
                            ch: ch)
                }
                
            case let .fullyConnected(ch, _):
                return (w: 1, h: 1, ch: ch)
            case let .maxPooling(_, stride):
                return (w: Int(ceil(Float(input.w) / Float(stride.0))),
                        h: Int(ceil(Float(input.h) / Float(stride.1))),
                        ch: input.ch)
            default:
                return input
            }
        }
        
        var name: String {
            switch self {
            case .convolution: return "conv"
            case .fullyConnected: return "fully"
            case .maxPooling: return "maxPool"
            case .softmax: return "softmax"
            }
        }
        
        var summary: String {
            switch self {
            case let .convolution(ch, w, h, stride, padding, activation):
                return "(\(ch), \(w), \(h)), \(stride), \(padding.summary), \(activation.summary)"
            case let .fullyConnected(ch, activation):
                return "\(ch), \(activation.summary)"
            case let .maxPooling(size, stride):
                return "\(size), \(stride)"
            case .softmax:
                return ""
            }
        }
    }
    
    enum Padding {
        case same, valid
        
        var summary: String {
            switch self {
            case .same: return "same"
            case .valid: return "valid"
            }
        }
    }
    
    enum Activation {
        case linear
        case relu
        
        func makeNeuron(device: MTLDevice) -> MPSCNNNeuron? {
            switch self {
            case .linear: return nil
            case .relu: return MPSCNNNeuronReLU(device: device, a: 0)
            }
        }
        
        var summary: String {
            switch self {
            case .linear: return ""
            case .relu: return "relu"
            }
        }
    }
}
