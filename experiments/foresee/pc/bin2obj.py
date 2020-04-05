import os
import numpy as np


def bin2obj(path, output_path):

    pts = np.fromfile(open(path, 'rb'), np.single).reshape([-1, 4])

    f = open(output_path, 'w')

    for i in range(pts.shape[0]):
        f.write('v %f %f %f %f %f %f\n' % (pts[i][0], pts[i][1], pts[i][2], pts[i][3], pts[i][3], pts[i][3]))
        
    f.close


if __name__ == "__main__":
    src_path = "pseudo-lidar/foresee/training/000039.bin"
    output_path = './' + os.path.basename(src_path).split('.')[0] + '.obj'
    bin2obj(src_path, output_path)
