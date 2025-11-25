"""
substrate_emergence_animation.py

Blender Python script for "How Constraints Create Bound States" animation.

Usage in Blender:
1. Open Blender
2. Delete default cube/camera/light
3. Open Scripting workspace
4. Load this script
5. Run script
6. Set output settings (resolution, frame rate, format)
7. Render animation (Render > Render Animation)

Duration: 30 seconds at 30 fps = 900 frames
Style: Clean geometric, teal/orange color scheme
"""

import bpy
import math
import mathutils

# =============================================================================
# Configuration
# =============================================================================

# Animation parameters
FPS = 30
DURATION = 30  # seconds
TOTAL_FRAMES = FPS * DURATION  # 900 frames

# Lattice parameters
LATTICE_SIZE = 7  # 7x7x7 grid
LATTICE_SPACING = 1.5  # units between nodes
NODE_RADIUS = 0.1
LINK_RADIUS = 0.03

# Color scheme (Ben's preference: teal + orange)
COLOR_NODES = (0.8, 0.8, 0.8, 1.0)  # Light gray
COLOR_CONSTRAINTS = (0.0, 0.7, 0.7, 1.0)  # Teal
COLOR_MATTER = (1.0, 0.5, 0.1, 1.0)  # Orange
COLOR_BOUND_STATE = (1.0, 0.6, 0.0, 1.0)  # Amber

# Material settings
METALLIC = 0.3
ROUGHNESS = 0.4

# Animation timing (frame numbers)
PHASE_1_START = 0      # Lattice appears
PHASE_1_END = 150      # 5 seconds

PHASE_2_START = 150    # Constraints appear
PHASE_2_END = 300      # 10 seconds

PHASE_3_START = 300    # Matter appears
PHASE_3_END = 450      # 15 seconds

PHASE_4_START = 450    # Settling/binding
PHASE_4_END = 750      # 25 seconds

PHASE_5_START = 750    # Camera pullback
PHASE_5_END = 900      # 30 seconds


# =============================================================================
# Utility functions
# =============================================================================

def clear_scene():
    """Delete all mesh objects in scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def create_material(name, color, metallic=0.3, roughness=0.4, emission=0.0):
    """Create a material with given properties."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Add nodes
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_emission = nodes.new(type='ShaderNodeEmission')
    node_mix = nodes.new(type='ShaderNodeMixShader')
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    
    # Set properties
    node_bsdf.inputs['Base Color'].default_value = color
    node_bsdf.inputs['Metallic'].default_value = metallic
    node_bsdf.inputs['Roughness'].default_value = roughness
    
    node_emission.inputs['Color'].default_value = color
    node_emission.inputs['Strength'].default_value = emission
    
    node_mix.inputs['Fac'].default_value = emission
    
    # Link nodes
    links.new(node_bsdf.outputs['BSDF'], node_mix.inputs[1])
    links.new(node_emission.outputs['Emission'], node_mix.inputs[2])
    links.new(node_mix.outputs['Shader'], node_output.inputs['Surface'])
    
    return mat


def create_sphere(location, radius, material):
    """Create a sphere at given location."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location, segments=16, ring_count=8)
    obj = bpy.context.active_object
    obj.data.materials.append(material)
    return obj


def create_cylinder(start, end, radius, material):
    """Create a cylinder between two points."""
    direction = mathutils.Vector(end) - mathutils.Vector(start)
    length = direction.length
    center = (mathutils.Vector(start) + mathutils.Vector(end)) / 2
    
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length, location=center)
    obj = bpy.context.active_object
    
    # Align cylinder to direction
    direction.normalize()
    up = mathutils.Vector((0, 0, 1))
    if direction != up and direction != -up:
        rotation_quat = up.rotation_difference(direction)
        obj.rotation_euler = rotation_quat.to_euler()
    elif direction == -up:
        obj.rotation_euler = (math.pi, 0, 0)
    
    obj.data.materials.append(material)
    return obj


def animate_fade_in(obj, start_frame, end_frame):
    """Animate object fading in by scaling from 0 to 1."""
    obj.scale = (0.001, 0.001, 0.001)
    obj.keyframe_insert(data_path="scale", frame=start_frame)
    
    obj.scale = (1.0, 1.0, 1.0)
    obj.keyframe_insert(data_path="scale", frame=end_frame)


def animate_position(obj, start_pos, end_pos, start_frame, end_frame):
    """Animate object moving from start to end position."""
    obj.location = start_pos
    obj.keyframe_insert(data_path="location", frame=start_frame)
    
    obj.location = end_pos
    obj.keyframe_insert(data_path="location", frame=end_frame)


# =============================================================================
# Scene setup
# =============================================================================

def setup_scene():
    """Configure scene settings."""
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 0
    scene.frame_end = TOTAL_FRAMES
    
    # Set background to dark
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg_node = world.node_tree.nodes['Background']
    bg_node.inputs['Color'].default_value = (0.02, 0.02, 0.02, 1.0)
    bg_node.inputs['Strength'].default_value = 1.0


def setup_camera():
    """Create and position camera."""
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_obj = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj
    
    # Initial position (close view)
    center = LATTICE_SIZE * LATTICE_SPACING / 2
    camera_obj.location = (center, center - 15, center + 5)
    camera_obj.rotation_euler = (math.radians(75), 0, 0)
    camera_obj.keyframe_insert(data_path="location", frame=0)
    camera_obj.keyframe_insert(data_path="rotation_euler", frame=0)
    
    # Final position (pulled back)
    camera_obj.location = (center, center - 20, center + 8)
    camera_obj.rotation_euler = (math.radians(70), 0, 0)
    camera_obj.keyframe_insert(data_path="location", frame=PHASE_5_END)
    camera_obj.keyframe_insert(data_path="rotation_euler", frame=PHASE_5_END)


def setup_lighting():
    """Create three-point lighting."""
    # Key light
    bpy.ops.object.light_add(type='AREA', location=(10, -10, 15))
    key = bpy.context.active_object
    key.data.energy = 200
    key.data.size = 5
    
    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-10, -5, 10))
    fill = bpy.context.active_object
    fill.data.energy = 100
    fill.data.size = 5
    
    # Rim light
    bpy.ops.object.light_add(type='AREA', location=(5, 5, 5))
    rim = bpy.context.active_object
    rim.data.energy = 150
    rim.data.size = 4


# =============================================================================
# Build animation elements
# =============================================================================

def create_lattice_nodes(mat_nodes):
    """Create lattice grid of nodes."""
    nodes = []
    for i in range(LATTICE_SIZE):
        for j in range(LATTICE_SIZE):
            for k in range(LATTICE_SIZE):
                x = i * LATTICE_SPACING
                y = j * LATTICE_SPACING
                z = k * LATTICE_SPACING
                
                node = create_sphere((x, y, z), NODE_RADIUS, mat_nodes)
                node.name = f"Node_{i}_{j}_{k}"
                
                # Animate fade in during phase 1
                delay = (i + j + k) * 3  # Stagger appearance
                animate_fade_in(node, PHASE_1_START + delay, PHASE_1_START + 50 + delay)
                
                nodes.append((node, i, j, k))
    
    return nodes


def create_constraint_links(nodes, mat_constraints):
    """Create constraint links between neighboring nodes."""
    links = []
    
    for node, i, j, k in nodes:
        # Link to neighbors (only forward to avoid duplicates)
        neighbors = [
            (i+1, j, k),
            (i, j+1, k),
            (i, j, k+1),
        ]
        
        for ni, nj, nk in neighbors:
            if ni < LATTICE_SIZE and nj < LATTICE_SIZE and nk < LATTICE_SIZE:
                # Find neighbor node
                neighbor = None
                for n, ii, jj, kk in nodes:
                    if ii == ni and jj == nj and kk == nk:
                        neighbor = n
                        break
                
                if neighbor:
                    start = node.location
                    end = neighbor.location
                    link = create_cylinder(start, end, LINK_RADIUS, mat_constraints)
                    link.name = f"Link_{i}_{j}_{k}_to_{ni}_{nj}_{nk}"
                    
                    # Animate fade in during phase 2
                    delay = (i + j + k) * 2
                    animate_fade_in(link, PHASE_2_START + delay, PHASE_2_START + 80 + delay)
                    
                    links.append(link)
    
    return links


def create_matter_particles(mat_matter):
    """Create matter particles that will settle into bound state."""
    particles = []
    center = LATTICE_SIZE * LATTICE_SPACING / 2
    
    # Create several particles at random positions
    num_particles = 8
    for idx in range(num_particles):
        # Random starting position (spread out)
        angle = idx * (2 * math.pi / num_particles)
        radius_start = center * 0.8
        x_start = center + radius_start * math.cos(angle)
        y_start = center + radius_start * math.sin(angle)
        z_start = center + (idx - num_particles/2) * 0.3
        
        # Final position (clustered in center)
        radius_end = center * 0.15
        x_end = center + radius_end * math.cos(angle) * 0.3
        y_end = center + radius_end * math.sin(angle) * 0.3
        z_end = center
        
        particle = create_sphere((x_start, y_start, z_start), NODE_RADIUS * 2.5, mat_matter)
        particle.name = f"Particle_{idx}"
        
        # Fade in during phase 3
        animate_fade_in(particle, PHASE_3_START + idx * 10, PHASE_3_START + 50 + idx * 10)
        
        # Move to bound state during phase 4
        animate_position(
            particle,
            (x_start, y_start, z_start),
            (x_end, y_end, z_end),
            PHASE_4_START,
            PHASE_4_END
        )
        
        particles.append(particle)
    
    return particles


def create_text_overlays():
    """Create text overlays for each phase."""
    texts = [
        ("Start with a quantum substrate", PHASE_1_START + 30, PHASE_1_END - 30),
        ("Local constraints couple nearby sites", PHASE_2_START + 30, PHASE_2_END - 30),
        ("Add matter degrees of freedom", PHASE_3_START + 30, PHASE_3_END - 30),
        ("Watch a stable pattern emerge", PHASE_4_START + 30, PHASE_4_END - 30),
        ("A bound state—from simple rules", PHASE_5_START + 30, PHASE_5_END - 30),
    ]
    
    for text_str, start_frame, end_frame in texts:
        bpy.ops.object.text_add(location=(0, 0, 0))
        text_obj = bpy.context.active_object
        text_obj.data.body = text_str
        text_obj.data.align_x = 'CENTER'
        text_obj.data.size = 0.5
        
        # Position in screen space (would need camera tracking in real implementation)
        # For now just make it visible/invisible
        text_obj.hide_render = True
        text_obj.keyframe_insert(data_path="hide_render", frame=start_frame - 1)
        
        text_obj.hide_render = False
        text_obj.keyframe_insert(data_path="hide_render", frame=start_frame)
        text_obj.keyframe_insert(data_path="hide_render", frame=end_frame)
        
        text_obj.hide_render = True
        text_obj.keyframe_insert(data_path="hide_render", frame=end_frame + 1)


# =============================================================================
# Main execution
# =============================================================================

def main():
    """Run full animation setup."""
    print("=" * 70)
    print("SUBSTRATE EMERGENCE ANIMATION - Blender Script")
    print("=" * 70)
    
    # Clear scene
    print("Clearing scene...")
    clear_scene()
    
    # Setup
    print("Setting up scene...")
    setup_scene()
    setup_camera()
    setup_lighting()
    
    # Create materials
    print("Creating materials...")
    mat_nodes = create_material("Mat_Nodes", COLOR_NODES, METALLIC, ROUGHNESS)
    mat_constraints = create_material("Mat_Constraints", COLOR_CONSTRAINTS, METALLIC, ROUGHNESS, emission=0.1)
    mat_matter = create_material("Mat_Matter", COLOR_MATTER, METALLIC, ROUGHNESS, emission=0.2)
    
    # Build elements
    print("Creating lattice nodes...")
    nodes = create_lattice_nodes(mat_nodes)
    print(f"  Created {len(nodes)} nodes")
    
    print("Creating constraint links...")
    links = create_constraint_links(nodes, mat_constraints)
    print(f"  Created {len(links)} links")
    
    print("Creating matter particles...")
    particles = create_matter_particles(mat_matter)
    print(f"  Created {len(particles)} particles")
    
    print("Creating text overlays...")
    create_text_overlays()
    
    print("\n" + "=" * 70)
    print("ANIMATION SETUP COMPLETE")
    print(f"Total frames: {TOTAL_FRAMES} ({DURATION} seconds at {FPS} fps)")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Set output format: Properties > Output > Format (PNG/MP4)")
    print("2. Set resolution: Properties > Output > Resolution")
    print("3. Render: Render > Render Animation (or F12)")
    print("=" * 70)


if __name__ == "__main__":
    main()