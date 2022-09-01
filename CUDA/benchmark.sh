#!/bin/bash
pixels=(1000)
# pixels=(6000 7000 8000 9000 10000)

for pixel_count in "${pixels[@]}"
do
    echo "quadric.ppm" >> out.txt
   ./raytrace $pixel_count $pixel_count ../examples/quadric.scene out_$pixel_count.ppm >> out.txt
done