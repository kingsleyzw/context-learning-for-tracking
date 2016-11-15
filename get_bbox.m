function bbox=get_bbox(detection)

bbox=[detection(:,[4,5]) detection(:,[4,5])+detection(:,[6,7])];