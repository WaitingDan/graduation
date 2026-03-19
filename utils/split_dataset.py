import os
import random
import shutil
import argparse
import json
from collections import defaultdict

random.seed(42)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_group_key(filename):
    stem = os.path.splitext(filename)[0]
    for token in ('__', '_', '-', ' '):
        if token in stem:
            prefix = stem.split(token)[0].strip()
            if prefix:
                return prefix
    return stem


def should_keep_class(class_name, include_keywords=None, exclude_keywords=None):
    low = class_name.lower()

    if include_keywords:
        if not any(k.lower() in low for k in include_keywords):
            return False

    if exclude_keywords:
        if any(k.lower() in low for k in exclude_keywords):
            return False

    return True


def split_dataset(
    origin_dataset,
    train_dir,
    val_dir,
    test_dir,
    split_rate_val=0.2,
    split_rate_test=0.1,
    grouped_split=True,
    include_keywords=None,
    exclude_keywords=None,
):

    classes = [d for d in os.listdir(origin_dataset) if os.path.isdir(os.path.join(origin_dataset, d))]

    for cla in classes:
        if not should_keep_class(cla, include_keywords=include_keywords, exclude_keywords=exclude_keywords):
            print(f"skip class: {cla}")
            continue

        cla_path = os.path.join(origin_dataset, cla)
        images = [f for f in os.listdir(cla_path) if os.path.isfile(os.path.join(cla_path, f))]

        num = len(images)

        if grouped_split:
            groups = defaultdict(list)
            for img in images:
                gid = parse_group_key(img)
                groups[gid].append(img)

            group_keys = list(groups.keys())
            random.shuffle(group_keys)

            total_groups = len(group_keys)
            val_group_num = int(total_groups * split_rate_val)
            test_group_num = int(total_groups * split_rate_test)

            val_groups = set(group_keys[:val_group_num])
            test_groups = set(group_keys[val_group_num:val_group_num + test_group_num])

            val_images = [img for gid in val_groups for img in groups[gid]]
            test_images = [img for gid in test_groups for img in groups[gid]]
            train_images = [img for gid in group_keys if gid not in val_groups and gid not in test_groups for img in groups[gid]]
        else:
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
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine', help='dataset subdir under project root')
    parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--test_rate', type=float, default=0.1)
    parser.add_argument('--no_grouped_split', action='store_true', help='disable grouped split by source id prefix')
    parser.add_argument('--include_keywords', nargs='*', default=None, help='keep classes containing any keyword')
    parser.add_argument('--exclude_keywords', nargs='*', default=None, help='drop classes containing any keyword')

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

    split_dataset(
        origin_dataset,
        train_dir,
        val_dir,
        test_dir,
        split_rate_val=args.val_rate,
        split_rate_test=args.test_rate,
        grouped_split=not args.no_grouped_split,
        include_keywords=args.include_keywords,
        exclude_keywords=args.exclude_keywords,
    )
    # 自动生成 class_indices.json（基于 train 目录的类顺序）
    try:
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        class_dict = {str(i): name for i, name in enumerate(classes)}
        out_path = os.path.join(root_dir, 'class_indices.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(class_dict, f, ensure_ascii=False, indent=4)
        print('Wrote class indices to', out_path)
    except Exception as e:
        print('Failed to write class_indices.json:', e)


if __name__ == '__main__':
    main()