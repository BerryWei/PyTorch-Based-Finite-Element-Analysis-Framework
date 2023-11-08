// 定義模型參數（SI 單位）
L = 0.20;           // 梁的長度 (單位: m)
a = 0.01;           // 梁橫截面的一半高度 (單位: m)
b = 0.001;          // 梁的寬度 (單位: m)
ne_x = 41;          // x方向的元素數量+1
ne_y = 5;           // y方向的元素數量
e_size_x = L / ne_x;  // x方向上每個元素的尺寸 (單位: m)
e_size_y = 2 * a / ne_y;  // y方向上每個元素的尺寸 (單位: m)

// 定義矩形的四個頂點
Point(1) = {0, -a, 0, e_size_x};
Point(2) = {L, -a, 0, e_size_x};
Point(3) = {L, a, 0, e_size_y};
Point(4) = {0, a, 0, e_size_y};

// 定義矩形的邊界線段
Line(1) = {1, 2};
Line(2) = {2, 3};  // 這是右邊的邊界
Line(3) = {3, 4};
Line(4) = {4, 1};

// 創建曲線迴圈和平面面
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// 定義線段的細分方式
Transfinite Line {1, 3} = ne_x;  // x方向
Transfinite Line {2, 4} = ne_y;  // y方向

// 指定平面面的細分方式
Transfinite Surface {1};
Recombine Surface {1};

// 定義物理群組（只保留2D元素）
Physical Surface("beam") = {1};

// 生成 2D 網格
Mesh 2;

