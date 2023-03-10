

[200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

[695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)


------


[200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

### 思路一：深度优先遍历DFS

- 目标是找到矩阵中 “岛屿的数量” ，上下左右相连的 1 都被认为是连续岛屿。
- dfs方法： 设目前指针指向一个岛屿中的某一点 (i, j)，寻找包括此点的岛屿边界。
	1. 从 (i, j) 向此点的上下左右 (i+1,j),(i-1,j),(i,j+1),(i,j-1) 做深度搜索。
	2. 终止条件：
		- (i, j) 越过矩阵边界;
		- grid[i][j] == 0，代表此分支已越过岛屿边界。
	3. 搜索岛屿的同时，执行 grid[i][j] = '0'，即将岛屿所有节点删除，以免之后重复搜索相同岛屿。

#### 主循环：

遍历整个矩阵，当遇到 grid[i][j] == '1' 时，从此点开始做深度优先搜索 dfs，岛屿数 count + 1 且在深度优先搜索中删除此岛屿。

- 最终返回岛屿数 count 即可。


``` go
func numIslands(grid [][]byte) int {
	count := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == '1' {
				dfs(grid, i, j)
				count++
			}
		}
	}
	return count
}
func dfs(grid [][]byte, i, j int) {
	if (i < 0 || j < 0) || i >= len(grid) || j >= len(grid[0]) || grid[i][j] == '0' {
		return
	}
	grid[i][j] = '0'
	dfs(grid, i+1, j)
	dfs(grid, i-1, j)
	dfs(grid, i, j-1)
	dfs(grid, i, j+1)
}
```


``` go
func numIslands(grid [][]byte) int {
	count := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == '1' {
				dfs(grid, i, j)
				count++
			}
		}
	}
	return count
}
func dfs(grid [][]byte, i, j int) {
	if (i < 0 || j < 0) || i >= len(grid) || j >= len(grid[0]) || grid[i][j] != '1' {
		return
	}
	grid[i][j] = '#'
	dfs(grid, i-1, j)
	dfs(grid, i+1, j)
	dfs(grid, i, j-1)
	dfs(grid, i, j+1)
}
```

#### 闭包

``` go
func numIslands(grid [][]byte) int {
	var dfs func(grid [][]byte, i, j int)
	dfs = func(grid [][]byte, i, j int) {
		if (i < 0 || j < 0) || i >= len(grid) || j >= len(grid[0]) || grid[i][j] == '0' {
			return
		}
		grid[i][j] = '0'
		dfs(grid, i+1, j)
		dfs(grid, i-1, j)
		dfs(grid, i, j-1)
		dfs(grid, i, j+1)
	}
	count := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == '1' {
				dfs(grid, i, j)
				count++
			}
		}
	}
	return count
}
```


[695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

``` go
func maxAreaOfIsland(grid [][]int) int {
	max_area := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == 1 {
				max_area = max(max_area, dfs(grid, i, j))
			}
		}
	}
	return max_area
}
func dfs(grid [][]int, i, j int) int {
	if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[0]) || grid[i][j] == 0 {
		return 0
	}
	area := 1
	grid[i][j] = 0
	area += dfs(grid, i+1, j)
	area += dfs(grid, i-1, j)
	area += dfs(grid, i, j+1)
	area += dfs(grid, i, j-1)
	return area
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```