// Set the sizes
sizeX = 20;
sizeY = 20;
sizeZ = 20;
numDivisions_X = 10;
numDivisions_Y = 10;
numDivisions_Z = 10; // Since sizeZ is much smaller, you might want only one layer in the Z direction.

// Calculate mesh size based on the number of divisions
meshSizeX = sizeX / numDivisions_X;
meshSizeY = sizeY / numDivisions_Y;
meshSizeZ = sizeZ / numDivisions_Z; // This should be equal to sizeZ if only one division is needed.

// Ensure that the divisions result in cubic elements
// The mesh sizes in each direction should be approximately equal for true cubic elements.
// Adjust numDivisions as needed to achieve this.

// Create points with mesh size specification
Point(1) = {0, 0, 0, meshSizeX};
Point(2) = {sizeX, 0, 0, meshSizeX};
Point(3) = {sizeX, sizeY, 0, meshSizeY};
Point(4) = {0, sizeY, 0, meshSizeY};
Point(5) = {0, 0, sizeZ, meshSizeZ};
Point(6) = {sizeX, 0, sizeZ, meshSizeZ};
Point(7) = {sizeX, sizeY, sizeZ, meshSizeZ};
Point(8) = {0, sizeY, sizeZ, meshSizeZ};

// Define lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// Define surfaces
Line Loop(13) = {1, 2, 3, 4};
Plane Surface(14) = {13};
Line Loop(15) = {5, 6, 7, 8};
Plane Surface(16) = {15};
Line Loop(17) = {1, 10, -5, -9};
Plane Surface(17) = {17};
Line Loop(18) = {2, 11, -6, -10};
Plane Surface(18) = {18};
Line Loop(19) = {3, 12, -7, -11};
Plane Surface(19) = {19};
Line Loop(20) = {4, 9, -8, -12};
Plane Surface(20) = {20};

// Volume
Surface Loop(21) = {14, 16, 17, 18, 19, 20};
Volume(22) = {21};

// The rest of your script remains unchanged until you get to the transfinite lines and volumes

// Transfinite lines - discretize the edges
Transfinite Line {1, 2} = numDivisions_X + 1 Using Progression 1;
Transfinite Line {3, 4} = numDivisions_Y + 1 Using Progression 1;
Transfinite Line {5, 6} = numDivisions_X + 1 Using Progression 1;
Transfinite Line {7, 8} = numDivisions_Y + 1 Using Progression 1;
Transfinite Line {9, 11} = numDivisions_Z + 1 Using Progression 1;
Transfinite Line {10, 12} = numDivisions_Z + 1 Using Progression 1;

// Transfinite surfaces with recombination into quadrilateral elements
Transfinite Surface {14, 16, 17, 18, 19, 20};
Recombine Surface {14, 16, 17, 18, 19, 20};

// Apply transfinite algorithm to the volume and mesh the volume with hex elements
Transfinite Volume {22};
Mesh.Algorithm3D = 1; // Delaunay for 3D mesh

// Delete existing 1D and 2D mesh before generating the 3D mesh
Delete { Point{1:8}; Line{1:12}; Surface{14:20}; }

// Generate the 3D mesh
Mesh 3;

// Delete the 1D and 2D elements to keep only the 3D elements
Delete { Line{1:12}; Surface{14:20}; }

// Now renumber the mesh to ensure 3D elements start at 1
Renumber Mesh;

// Generate the 3D mesh
Mesh 3;//+
Recombine Surface {19, 20, 14, 16, 17, 18};
//+
Recombine Surface {20, 19, 14, 16, 17, 18};
