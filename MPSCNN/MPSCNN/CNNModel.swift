//
//  CNNModel.swift
//  MPSCNN
//
//  Created by M.Ike on 2017/04/30.
//  Copyright © 2017年 M.Ike. All rights reserved.
//

import Foundation

protocol CNNModel {
    var inputFormat: CNN.Format { get }
    func makeModel() -> [CNN.Layer]
}

extension CNNModel {
    func conv2D(_ name: String,
                _ channel: Int,
                kernel: (Int, Int),
                stride: (Int, Int) = (1, 1),
                pad: CNN.Layer.Padding = .valid,
                activation: CNN.Layer.Activation = .linear,
                input: String = "", output: String = "") -> CNN.Layer {
        return CNN.Layer(name: name,
                         type: .convolution(ch: channel, w: kernel.0, h: kernel.1,
                                            stride: stride, padding: pad, activation: activation),
                         inputImage: input, outputImage: output, outputOffset: 0)
    }
    
    func maxPooling2D(_ name: String,
                      kernel: (Int, Int),
                      stride: (Int, Int),
                      input: String = "", output: String = "") -> CNN.Layer {
        return CNN.Layer(name: name,
                         type: .maxPooling(size: kernel, stride: stride),
                         inputImage: input, outputImage: output, outputOffset: 0)
    }
    
    func averagePooling2D(_ name: String,
                          kernel: (Int, Int),
                          stride: (Int, Int),
                          input: String = "", output: String = "") -> CNN.Layer {
        return CNN.Layer(name: name,
                         type: .averagePooling(size: kernel, stride: stride),
                         inputImage: input, outputImage: output, outputOffset: 0)
    }
    
    func fully(_ name: String,
               _ channel: Int,
               activation: CNN.Layer.Activation = .linear,
               input: String = "", output: String = "") -> CNN.Layer {
        return CNN.Layer(name: name,
                         type: .fullyConnected(ch: channel, activation: activation),
                         inputImage: input, outputImage: output, outputOffset: 0)
    }
    
    func softmax(_ name: String, input: String = "", output: String = "") -> CNN.Layer {
        return CNN.Layer(name: name,
                         type: .softmax,
                         inputImage: input, outputImage: output, outputOffset: 0)
    }
}
