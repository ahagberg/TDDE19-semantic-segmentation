
import glob
import os

EXTENSIONS = ["jpg", "jpeg", "png"]

def glob_match_image_files(path):
    """
    Match all jpg, png and jpeg files for images.
    """
    return sum([glob.glob(os.path.join(path, "*%s" % extension)) for extension in EXTENSIONS], [])

def remove_extension(path):
    """
    Removes image extensions from paths.
    """
    for extension in EXTENSIONS:
        path = path.replace(".%s" % extension, "")
    return path

def get_img_annot_pairs_from_paths(images_path , segs_path):
    """
    Returns a list of pairs of paths to images and their annotations.
    """
    imagepaths = glob_match_image_files(images_path)
    annotpaths  =  glob_match_image_files(segs_path)
    annotnames = [remove_extension(os.path.basename(path)) for path in annotpaths]

    ret = []

    for imgpath in imagepaths:

        imgname = remove_extension(os.path.basename(imgpath))

        try:
            index = annotnames.index(imgname)
        except:
            print("Image: '%s', does not have an annotation!" % imgpath)
            continue

        segpath = annotpaths[index]

        ret.append((imgpath , segpath))

    return ret

###########################################################################
###                    TEST THE IMPLEMENTATION HERE
###########################################################################

if __name__ == '__main__':

    pairs = get_img_annot_pairs_from_paths("./datasets/CamVid/train", "./datasets/CamVid/trainannot")

    print(pairs)
