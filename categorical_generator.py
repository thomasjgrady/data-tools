# A script to generate fake categorical data consisting of colored polygons.
# Each k-tuple of 2-tuples of the number of sides and color
# ((n1,c1), (n2,c2), ... , (nk, ck)) represents a unique label, allowing for
# the creation of datasets with a large number of classes.

from argparse import ArgumentParser
from math import sin, cos
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

import colorsys
import json
import numpy as np
import os
import random
import shutil
import sys

parser = ArgumentParser()
parser.add_argument('--width', '-w', type=int, default=256)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--group_size', '-k', type=int, default=1)
parser.add_argument('--max_groups', '-mg', type=int, default=3)
parser.add_argument('--max_sides', '-ms', type=int, default=13)
parser.add_argument('--num_colors', '-nc', type=int, default=10)
parser.add_argument('--num_images', '-n', type=int, default=1000)
parser.add_argument('--output_dir', '-o', type=Path, default=Path('./data'))
parser.add_argument('--scale_factor', '-sf', type=float, default=10)
parser.add_argument('--translation_factor', '-tf', type=float, default=4)

args = parser.parse_args()
w = args.width
h = args.height
n = args.num_images
group_size = args.group_size
max_groups = args.max_groups
max_sides = args.max_sides
num_colors = args.num_colors
output_dir = args.output_dir
scale_factor = args.scale_factor
translation_factor = args.translation_factor

num_classes = num_colors * (max_sides-3) * group_size

if output_dir.exists():
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

colors_hsv = [(k/num_colors, 1, 1) for k in range(num_colors)]
colors_rgb = [colorsys.hsv_to_rgb(*c) for c in colors_hsv]
colors = [(round(c[0] * 255), round(c[1] * 255), round(c[2] * 255)) for c in colors_rgb]

label_shape = []
for k in range(group_size):
    label_shape.append(max_sides-2)
    label_shape.append(num_colors)

print(f'Generating dataset with {np.prod(label_shape)} classes.')

for i in tqdm(range(n)):

    n_groups = random.randint(1, max_groups)

    img = Image.new('RGB', (w, h), 'black')
    draw = ImageDraw.Draw(img)
    
    label = np.zeros(label_shape)

    for j in range(n_groups):

        n_sides = [random.randint(3, max_sides) for _ in range(group_size)]

        vertices = []
        for ns in n_sides:
            verts = np.exp([2*complex(real=0.0, imag=1.0)*np.pi*k/ns for k in range(ns)])
            vertices.append(np.array([[c.real, c.imag] for c in verts]))

        # Rotate the vertices a random amount
        thetas = [random.random() * 2*np.pi for _ in range(group_size)]
        Rs = [np.array([[cos(t), -sin(t)], [sin(t), cos(t)]]) for t in thetas]
        vertices = [v.T for v in vertices]
        vertices = [R @ v for R, v in zip(Rs, vertices)]
        
        # Scale the vertices a random amount
        base_scale = float(min(w, h) // scale_factor)
        scales = [random.random() * 2.0 + 1.0 for _ in range(group_size)]
        vertices = [base_scale * s * v for s, v in zip(scales, vertices)]

        # Translate the vertices a random amount from the center
        base_translation = np.atleast_2d([w / 2, h / 2]).T
        translations = [np.atleast_2d(
            [random.random() * (w/(translation_factor/2)) - (w/translation_factor),
             random.random() * (h/(translation_factor/2)) - (h/translation_factor)]).T for _ in range(group_size)]
        vertices = [base_translation + t + v for t, v in zip(translations, vertices)]

        # Put the vertices back into the correct datatype etc
        vertices = [v.astype(np.int32).T for v in vertices]
        
        # Draw polygons
        cis = [random.randint(0, num_colors-1) for _ in range(group_size)]
        cs = [colors[j] for j in cis]

        for v, c in zip(vertices, cs):
            l = v.tolist()
            l = tuple([tuple(x) for x in l])
            draw.polygon(l, tuple(c))
        
        # Draw connections
        if group_size > 1:
            centers = [base_translation + t for t in translations]
            for k in range(group_size-1):
                c0 = centers[k]
                c1 = centers[k+1]
                l = ((c0[0,0], c0[1,0]), (c1[0,0], c1[1,0]))
                c = (255, 255, 255)
                draw.line(l, c, width=10)

        # Save the label
        index = [0 for _ in range(group_size*2)]
        for k in range(0, group_size*2, 2):
            ns = n_sides[k//2] - 3
            ci = cis[k//2]
            index[k] = ns
            index[k+1] = ci

        label[tuple(index)] = 1
    
    # Save the image
    img_output_path = output_dir.joinpath(Path(f'image_{i:06d}.png'))
    img.save(img_output_path)

    # Save the label
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
