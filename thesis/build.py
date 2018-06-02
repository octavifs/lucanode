#!/usr/bin/env python
#encoding: utf-8

import os
from subprocess import call


def isimage(filename):
    """Test if a filename has an image termination"""
    if (
        '.png' in filename or
        '.jpg' in filename or
        '.jpeg' in filename
    ):
        return True
    else:
        return False


def symlink(source, dest):
    try:
        os.symlink(source, dest)
    except OSError:
        pass


def unsymlink(link):
    try:
        os.remove(link)
    except OSError:
        pass

# Variables with binary location, bibliography and citation styles
pandoc_bin = '/usr/local/bin/pandoc'
bibliography = 'library.bib'
citation_style = 'acm-siggraph.csl'

# List markdown files in order
markdown_files = ['title.md']
chapter_files = sorted([dir[0] + '/' + file for dir in os.walk('.') for file in dir[2] if dir[0] is not '.' and '.md' in file])
markdown_files.extend(chapter_files)
markdown_files.append('bibliography.md')
# Get image paths and names
image_files = [dir[0] + '/' + file for dir in os.walk('.') for file in dir[2] if dir[0] is not '.' and isimage(file)]
image_names = ['./' + image.split('/')[-1] for image in image_files]
# Symlink images to current dir
for img_src, img_dest in zip(image_files, image_names):
    symlink(img_src, img_dest)

# Prepare command
pandoc_args = [
    pandoc_bin,
    '-s',
    '--mathjax',
    '--bibliography', bibliography,
    '--csl', citation_style,
    '--toc',
    '--number-sections',
    '--top-level-division=chapter',
    '-o', 'TFM.pdf'
]
pandoc_args.extend(markdown_files)

# Generate PDF
call(pandoc_args)

# Remove image symlinks
for img_name in image_names:
    unsymlink(img_name)

# Open the PDF
call(["/usr/bin/open", "TFM.pdf"])
