//
//  MPSImage+Array.swift
//  MPSCNN
//
//  Created by M.Ike on 2017/04/30.
//  Copyright © 2017年 M.Ike. All rights reserved.
//

import MetalPerformanceShaders
import Accelerate

extension MPSImage {
    func float16ToFloatArray() -> [Float] {
        let channel = 4
        let slice = (featureChannels + (channel - 1)) / channel
        let count = texture.width * texture.height * featureChannels
        
        var output = [Float](repeating: 0, count: count)
        autoreleasepool {
            var buf = [UInt16](repeating: 0, count: count)
            
            if featureChannels > 4 {
                for i in 0..<slice {
                    texture.getBytes(&(buf[height * width * channel * i]),
                                     bytesPerRow: MemoryLayout<UInt16>.size * width * channel,
                                     bytesPerImage: 0,
                                     from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                                     size: MTLSize(width: width, height: height, depth: 1)),
                                     mipmapLevel: 0,
                                     slice: i)
                }
            } else {
                texture.getBytes(&buf,
                                 bytesPerRow: MemoryLayout<UInt16>.size * width * channel,
                                 from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                                 size: MTLSize(width: width, height: height, depth: 1)),
                                 mipmapLevel: 0)
                
            }
            
            var vbuf16 = vImage_Buffer(data: &buf, height: 1, width: UInt(count),
                                       rowBytes: count * MemoryLayout<UInt16>.size)
            var vbuf32 = vImage_Buffer(data: &output, height: 1, width: UInt(count),
                                       rowBytes: count * MemoryLayout<Float>.size)
            vImageConvert_Planar16FtoPlanarF(&vbuf16, &vbuf32, 0)
        }
        
        return output
    }
}
