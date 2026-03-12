import os
import random
import shutil
import argparse

random.seed(42)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_dataset(origin_dataset, train_dir, val_dir, test_dir, split_rate_val=0.2, split_rate_test=0.1):

    classes = [d for d in os.listdir(origin_dataset) if os.path.isdir(os.path.join(origin_dataset, d))]

    for cla in classes:

        cla_path = os.path.join(origin_dataset, cla)
        images = [f for f in os.listdir(cla_path) if os.path.isfile(os.path.join(cla_path, f))]

        num = len(images)

        val_num = int(num * split_rate_val)
        test_num = int(num * split_rate_test)

        random.shuffle(images)

        val_images = images[:val_num]
        test_images = images[val_num:val_num + test_num]
        train_images = images[val_num + test_num:]

        for img in train_images:
            src = os.path.join(cla_path, img)
            dst = os.path.join(train_dir, cla)
            make_dir(dst)
            shutil.copy(src, os.path.join(dst, img))

        for img in val_images:
            src = os.path.join(cla_path, img)
            dst = os.path.join(val_dir, cla)
            make_dir(dst)
            shutil.copy(src, os.path.join(dst, img))

        for img in test_images:
            src = os.path.join(cla_path, img)
            dst = os.path.join(test_dir, cla)
            make_dir(dst)
            shutil.copy(src, os.path.join(dst, img))

        print(f"{cla} -> train:{len(train_images)} val:{len(val_images)} test:{len(test_images)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None, help='project root directory (defaults to two levels up)')
    parser.add_argument('--source', default='FGSCR', help='source folder name under dataset root')
    parser.add_argument('--dataset_subdir', default='dataset/ship_cls', help='dataset subdir under project root')
    parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--test_rate', type=float, default=0.1)

    args = parser.parse_args()

    if args.root is None:
        # default to project root relative to this file
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        root_dir = args.root

    dataset_root = os.path.join(root_dir, args.dataset_subdir)
    origin_dataset = os.path.join(dataset_root, args.source)

    train_dir = os.path.join(dataset_root, 'train')
    val_dir = os.path.join(dataset_root, 'val')
    test_dir = os.path.join(dataset_root, 'test')

    split_dataset(origin_dataset, train_dir, val_dir, test_dir, args.val_rate, args.test_rate)


if __name__ == '__main__':
    main()