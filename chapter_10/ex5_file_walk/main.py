import os
import filetype
import shutil


def collect_files(folder, outfolder, ext='pdf', use_magic=False):

    if not os.path.isdir(outfolder):
        print("Output directory not found! Exit.")
        exit()
    ext_pt = '.' + ext
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(folder)))

    for root, dirs, files in os.walk(folder):

        if len(files) > 0:
            for fname in files:
                src_file = os.path.join(root_path, root, fname)
                is_ext = False
                if use_magic:
                    kind = filetype.guess(src_file)
                    if kind is not None:
                        if kind.extension == ext:
                            is_ext = True
                else:
                    if src_file.endswith(ext):
                        is_ext = True

                if is_ext:
                    # copy file
                    fname = os.path.splitext(fname)[0]
                    # check if filename is in destination folder
                    while fname + ext_pt in os.listdir(outfolder):
                        fname += '.1'
                    dest_file = os.path.join(root_path, outfolder, fname + ext_pt)

                    shutil.copy(src_file, dest_file)


if __name__ == '__main__':
    folder = 'tmp/file_in'
    output_folder = 'tmp/file_out'
    collect_files(folder, output_folder, use_magic=False)


