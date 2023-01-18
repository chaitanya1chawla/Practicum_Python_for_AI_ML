import os
import sys
import shutil


def rename_imgs(folder, output_folder):

    if not os.path.isdir(output_folder):
        print("Output directory not found! Exit.")
        exit()

    # for root, dirs, files in os.walk(folder):
    root = os.path.dirname(os.path.dirname(os.path.abspath(folder)))
    files = os.listdir(folder)
    if len(files) > 0:
        for fname in files:
            if fname.lower().endswith('.jpeg') or fname.lower().endswith('.jpg'):
                # copy image in canonicalize form
                src_file = os.path.join(root, folder, fname)

                fname = os.path.splitext(fname)[0]
                # check if filename is in destination folder
                while fname + '.jpg' in os.listdir(output_folder):
                    fname += '.1'
                dest_file = os.path.join(root, output_folder, fname + '.jpg')

                shutil.copy(src_file, dest_file)
                # try:
                #     os.system(f'copy {src_file} {dest_file}')
                # except:
                #     os.system(f'cp {src_file} {dest_file}')


if __name__ == '__main__':
    folder = 'tmp/img_in'
    output_folder = 'tmp/img_out'
    rename_imgs(folder, output_folder)
