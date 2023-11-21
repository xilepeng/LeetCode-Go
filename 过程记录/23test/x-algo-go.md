


## [240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

```go
func searchMatrix(matrix [][]int, target int) bool {
	x, y := 0, len(matrix[0])-1
	for x < len(matrix) && y >= 0 {
		if matrix[x][y] == target {
			return true
		}
		if target < matrix[x][y] {
			y--
		} else {
			x++
		}
	}
	return false
}
```