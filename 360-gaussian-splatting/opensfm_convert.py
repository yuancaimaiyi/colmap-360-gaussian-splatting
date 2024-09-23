import os
import cv2 
import perspective_and_equirectangular.lib.Equirec2Perspec as E2P
import perspective_and_equirectangular.lib.Perspec2Equirec as P2E
import perspective_and_equirectangular.lib.multi_Perspec2Equirec as m_P2E
import glob
import argparse

def panorama2cube4(input_dir):
    base_dir = os.path.basename(input_dir.rstrip('/\\'))
    if not base_dir:
        base_dir = 'images'

    output_dir = os.path.join(input_dir, '..', base_dir + '_split')

    all_image = sorted(glob.glob(input_dir + '/*.*'))
    height, width = cv2.imread(all_image[0]).shape[:2]
    cube_size = int(width / 4)


    for index in range(len(all_image)):
        equ = E2P.Equirectangular(all_image[index])    # Load equirectangular image

        img = equ.GetPerspective(90, 0, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'front.jpg'
        cv2.imwrite(output1, img)

        img = equ.GetPerspective(90, 90, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output2 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'right.jpg'

        cv2.imwrite(output2, img)

        img = equ.GetPerspective(90, 180, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output3 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'back.jpg'

        cv2.imwrite(output3, img)

        img = equ.GetPerspective(90, 270, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output4 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'left.jpg'

        cv2.imwrite(output4, img)

def panorama2cube(input_dir):
    base_dir = os.path.basename(input_dir.rstrip('/\\'))
    if not base_dir:
        base_dir = 'images'

    all_image = sorted(glob.glob(input_dir + '/*.*'))
    height, width = cv2.imread(all_image[0]).shape[:2]
    cube_size = int(width / 4)

    for index in range(len(all_image)):
        equ = E2P.Equirectangular(all_image[index])    # Load equirectangular image

        out_img = input_dir + '/' + os.path.basename(all_image[index])
        img_0 = equ.GetPerspective(90, 0, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        img_right = equ.GetPerspective(90, 90, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        img_left = equ.GetPerspective(90, -90, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        img_back = equ.GetPerspective(90, 180, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)

        img = cv2.hconcat([img_left, img_0, img_right, img_back])
        cv2.imwrite(out_img, img)

def main():
    parser = argparse.ArgumentParser(description="Convert equirectangular panorama to cube map.")
    parser.add_argument("input_dir", type=str, help="Input directory containing equirectangular images.")
    parser.add_argument("--split", action='store_true', help="Split the panorama into 4 images (front, right, back, left)")

    args = parser.parse_args()

    if args.split:
        panorama2cube4(args.input_dir)
    else:
        panorama2cube(args.input_dir)

if __name__ == "__main__":
    main()