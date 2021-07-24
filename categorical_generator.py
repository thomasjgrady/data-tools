# A script to generate fake categorical data consisting of colored polygons
# Each tuple (num_sides, color) defines a unique class. The polygons are
# arbitrarily rotated, translated, and scaled based on user-defined arguments.

from argparse import ArgumentParser
from math import sin, cos
from pathlib import Path
from PIL import Image, ImageDraw

import colorsys
import json
import numpy as np
import os
import random
import shutil
import sys

parser = ArgumentParser()
parser.add_argument('--num_images', '-n', type=int, default=1000)
parser.add_argument('--width', '-w', type=int, default=256)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--max_sides', '-ms', type=int, default=13)
parser.add_argument('--max_polys', '-mp', type=int, default=3)
parser.add_argument('--num_colors', '-nc', type=int, default=10)
parser.add_argument('--scale_factor', '-sf', type=float, default=10)
parser.add_argument('--translation_factor', '-tf', type=float, default=4)
parser.add_argument('--output_dir', '-o', type=Path, default=Path('./data'))

args = parser.parse_args()
n = args.num_images
w = args.width
h = args.height
max_sides = args.max_sides
max_polys = args.max_polys
num_colors = args.num_colors
scale_factor = args.scale_factor
translation_factor = args.translation_factor
output_dir = args.output_dir

num_classes = num_colors * (max_sides-3)

if output_dir.exists():
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

colors_hsv = [(k/num_colors, 1, 1) for k in range(num_colors)]
colors_rgb = [colorsys.hsv_to_rgb(*c) for c in colors_hsv]
colors = [(round(c[0] * 255), round(c[1] * 255), round(c[2] * 255)) for c in colors_rgb]

for i in range(n):

    n_polys = random.randint(1, max_polys)
    n_sides = [random.randint(3, max_sides-1) for _ in range(n_polys)]

    vertices = []
    for ns in n_sides:
        verts = np.exp([2*complex(real=0.0, imag=1.0)*np.pi*k/ns for k in range(ns)])
        vertices.append(np.array([[c.real, c.imag] for c in verts]))


    # Rotate the vertices a random amount
    thetas = [random.random() * 2*np.pi for _ in range(n_polys)]
    Rs = [np.array([[cos(t), -sin(t)], [sin(t), cos(t)]]) for t in thetas]
    vertices = [v.T for v in vertices]
    vertices = [R @ v for R, v in zip(Rs, vertices)]
    
    # Scale the vertices a random amount
    base_scale = float(min(w, h) // scale_factor)
    scales = [random.random() * 2.0 + 1.0 for _ in range(n_polys)]
    vertices = [base_scale * s * v for s, v in zip(scales, vertices)]

    # Translate the vertices a random amount from the center
    base_translation = np.atleast_2d([w / 2, h / 2]).T
    translations = [np.atleast_2d(
        [random.random() * (w/(translation_factor/2)) - (w/translation_factor),
         random.random() * (h/(translation_factor/2)) - (h/translation_factor)]).T for _ in range(n_polys)]
    vertices = [base_translation + t + v for t, v in zip(translations, vertices)]

    # Put the vertices back into the correct datatype etc
    vertices = [v.astype(np.int32).T for v in vertices]

    # Draw the image
    img = Image.new('RGB', (w, h), 'black')
    draw = ImageDraw.Draw(img)

    cis = [random.randint(0, num_colors-1) for _ in range(n_polys)]
    cs = [colors[j] for j in cis]

    for v, c in zip(vertices, cs):
        l = v.tolist()
        l = tuple([tuple(x) for x in l])
        draw.polygon(l, tuple(c))

    img_output_path = output_dir.joinpath(Path(f'image_{i:06d}.png'))
    img.save(img_output_path)

    label = np.zeros(shape=(max_sides-3, num_colors))
    for v, ci in zip(vertices, cis):
        label[len(v)-3][ci] = 1

    label_output_path = output_dir.joinpath(Path(f'label_{i:06d}.txt'))
    label = label.flatten()
    np.savetxt(label_output_path, label)

meta_output_path = output_dir.joinpath('meta.txt')
with open(meta_output_path, 'w') as f:
    d0 = vars(args)
    d1 = {}
    for k, v in d0.items():
        v_out = v
        if issubclass(type(v), Path):
            v_out = str(v)
        d1[k] = v_out
    s = json.dumps(d1)
    f.write(s)