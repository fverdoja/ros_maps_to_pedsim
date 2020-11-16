#!/usr/bin/env python

import rospy
import yaml
import os.path

import numpy as np
import xml.etree.ElementTree as xml
import skimage.io as io

from xml.dom import minidom


def get_window(image, x, y):
    """
    Returns a window around a pixel.

    The windows is a 3x3 window centererd around pixel (x, y). If the pixel is
    close to the edges of the image, the window will be smaller, accordingly
    (e.g., the method will return only a 2x2 window for pixel (0, 0)).

        Parameters:
            image (array_like): an image from which the window is extracted
            x (int): x coordinate of the pixel
            y (int): y coordinate of the pixel

        Returns:
            window (array_like): a window around the pixel (x, y)
    """
    sz = image.shape
    assert (x >= 0 and x < sz[0] and y >= 0 and y <
            sz[1]), "Pixel indeces out of image bounds (%d, %d)" % (x, y)

    x_min = np.maximum(0, x-1)
    x_max = np.minimum(sz[0], x+2)
    y_min = np.maximum(0, y-1)
    y_max = np.minimum(sz[1], y+2)

    return image[x_min:x_max, y_min:y_max]


def add_waypoint(scenario, id, x, y, r):
    """Adds to a scenario a waypoint named 'id' in (x, y) with radius 'r'"""
    waypoint = xml.SubElement(scenario, 'waypoint')
    waypoint.set('id', str(id))
    waypoint.set('x', str(x))
    waypoint.set('y', str(y))
    waypoint.set('r', str(r))


def add_agent(scenario, x, y, waypoints, n=2, dx=0.5, dy=0.5):
    """Adds to a scenario n agents going from (x, y) through the waypoints"""
    agent = xml.SubElement(scenario, 'agent')
    agent.set('x', str(x))
    agent.set('y', str(y))
    agent.set('n', str(n))
    agent.set('dx', str(dx))
    agent.set('dy', str(dy))
    for w in waypoints:
        addwaypoint = xml.SubElement(agent, 'addwaypoint')
        addwaypoint.set('id', str(w[0]))


def add_waypoints_and_agent(scenario, waypoints):
    """Adds to a scenario a set of waypoints and an agent going through them"""
    for w in waypoints:
        add_waypoint(scenario, w[0], w[1], w[2], w[3])
    add_agent(scenario, waypoints[-1][1], waypoints[-1][2], waypoints)


def add_obstacle(scenario, x1, y1, x2, y2):
    """Adds to a scenario an obstacle going from (x1, y1) to (x2, y2)"""
    obstacle = xml.SubElement(scenario, 'obstacle')
    obstacle.set('x1', str(x1))
    obstacle.set('y1', str(y1))
    obstacle.set('x2', str(x2))
    obstacle.set('y2', str(y2))


def add_pixel_obstacle(scenario, x, y):
    """Adds to a scenario a 1x1 obstacle at location (x, y)"""
    add_obstacle(scenario, x, y, x, y)


def scenario_from_map(map_image, map_metadata):
    """
    Builds a pedsim scenario having obstacles to separate free space in the map
    from unknown and occupied space. Everything below 'free_thresh' (in the map
    metadata) is considered free space.

        Parameters:
            map_image (array_like): the map ternary image
            map_metadata (dictionary): the metadata extracted from the map YAML
                file

        Returns:
            scenario (ElementTree): a pedsim scenario as xml element tree
            map_walls (array_like): a binary image showing the locations on the
                map where obstacles have been placed
    """
    resolution = map_metadata['resolution']
    origin = map_metadata['origin']
    negate = map_metadata['negate']
    free_thresh = map_metadata['free_thresh'] * 255

    # ROS maps have white (255) as free space for visualization, colors need to
    # be inverted before comparing with thresholds (if negate == 0)
    if ~negate:
        map_binary = 255-map_image < free_thresh
    else:
        map_binary = map_image < free_thresh

    scenario = xml.Element('scenario')

    sz = map_binary.shape
    map_walls = np.zeros(sz, dtype=bool)

    # reduce the search space to only the area where there is free space
    x_free = np.nonzero(np.sum(map_binary, axis=1))[0]
    x_min = np.maximum(0, x_free[0]-1)
    x_max = np.minimum(sz[0], x_free[-1]+2)
    y_free = np.nonzero(np.sum(map_binary, axis=0))[0]
    y_min = np.maximum(0, y_free[0]-1)
    y_max = np.minimum(sz[1], y_free[-1]+2)

    for x in xrange(x_min, x_max):
        for y in xrange(y_min, y_max):
            is_free = map_binary[x, y]
            window = get_window(map_binary, x, y)
            if ~is_free and np.any(window) and np.any(~window):
                # conversion between world coordinates and pixel coordinates
                # (x and y coordinates are inverted, and y is also flipped)
                world_x = y * resolution
                world_y = - x * resolution - 2 * origin[1]

                add_pixel_obstacle(scenario, world_x, world_y)
                map_walls[x, y] = True

    return scenario, map_walls


def write_xml(tree, file_path, indent="  "):
    """Takes an xml tree and writes it to a file, indented"""
    indented_xml = minidom.parseString(
        xml.tostring(tree)).toprettyxml(indent=indent)

    with open(file_path, "w") as f:
        f.write(indented_xml)


if __name__ == '__main__':
    rospy.init_node('ros_maps_to_pedsim', anonymous=True)

    map_path = rospy.get_param("~map_path", ".")
    map_name = rospy.get_param("~map_name", "map.yaml")
    scenario_path = rospy.get_param("~scenario_path", ".")
    scenario_name = rospy.get_param("~scenario_name", "scene.xml")

    with open(os.path.join(map_path, map_name)) as file:
        map_metadata = yaml.safe_load(file)

    map_image = io.imread(os.path.join(map_path, map_metadata['image']))

    print("Loaded map in " + os.path.join(map_path, map_name)
          + " with metadata:")
    print(map_metadata)

    scenario, map_walls = scenario_from_map(map_image, map_metadata)

    # the next two lines add a group of agents moving through the scenario;
    # these waypoints work in my testing map, remove or substitute with yours
    waypoints = [
        ['w1', 18, 9, 0.2],
        ['w2', 20, 8, 0.2],
        ['w3', 21.5, 7.5, 0.2],
        ['w4', 18, 8.5, 0.2]
    ]
    add_waypoints_and_agent(scenario, waypoints)

    # uncomment for a visualization of where the obstacles have been placed
    # io.imsave(os.path.join(scenario_path, 'walls.png'), map_walls*255)

    print("Writing scene in " + os.path.join(scenario_path, scenario_name)
          + "...")

    write_xml(scenario, os.path.join(scenario_path, scenario_name))

    print("Done.")
