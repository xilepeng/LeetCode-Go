

[54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

[59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)

------

[54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1goyc6zkdssj31ef0eiwjn.jpg)

- 如果一条边从头遍历到底，则下一条边遍历的起点随之变化

- 选择不遍历到底，可以减小横向、竖向遍历之间的影响

- 轮迭代结束时，4条边的两端同时收窄 1

- 轮迭代所做的事情很清晰：遍历一个“圈”，遍历的范围收缩为内圈

- 层层向里处理，按顺时针依次遍历：上、右、下、左。

- 不再形成“环”了，就会剩下一行或一列，然后单独判断

### 四个边界

- 上边界 top : 0
- 下边界 bottom : matrix.length - 1
- 左边界 left : 0
- 右边界 right : matrix[0].length - 1

### 矩阵不一定是方阵
- top < bottom && left < right 是循环的条件
- 无法构成“环”了，就退出循环，退出时可能是这 3 种情况之一：
- top == bottom && left < right —— 剩一行
- top < bottom && left == right —— 剩一列
- top == bottom && left == right —— 剩一项（也是一行/列）

### 处理剩下的单行或单列
- 因为是按顺时针推入结果数组的，所以
- 剩下的一行，从左至右 依次推入结果数组
- 剩下的一列，从上至下 依次推入结果数组

### 代码

每个元素访问一次，时间复杂度 O(m*n)，m、n 分别是矩阵的行数和列数
空间复杂度 O(m*n)

``` go
func spiralOrder(matrix [][]int) []int {
    if len(matrix) == 0 {
        return []int{}
    }
    res := []int{}
    top, bottom, left, right := 0, len(matrix)-1, 0, len(matrix[0])-1
    
    for top < bottom && left < right {
        for i := left; i < right; i++ { res = append(res, matrix[top][i]) }
        for i := top; i < bottom; i++ { res = append(res, matrix[i][right]) }
        for i := right; i > left; i-- { res = append(res, matrix[bottom][i]) }
        for i := bottom; i > top; i-- { res = append(res, matrix[i][left]) }
        right--
        top++
        bottom--
        left++ 
    }
    if top == bottom {
        for i := left; i <= right; i++ { res = append(res, matrix[top][i]) }
    } else if left == right {
        for i := top; i <= bottom; i++ { res = append(res, matrix[i][left]) }
    } 
    return res
}
```

## 换一种遍历的策略：遍历到底

![2.png](http://ww1.sinaimg.cn/large/007daNw2ly1goyc7ez0y4j31l00fswk5.jpg)


- 循环的条件改为： top <= bottom && left <= right
- 每遍历一条边，下一条边遍历的起点被“挤占”，要更新相应的边界
- 值得注意的是，可能出现 在循环中途，不再满足循环的条件 ，即出现 top > bottom || left > right ，其中一对边界彼此交错了
- 这意味着此时所有项都遍历完了，如果没有及时 break ，就会重复遍历

### 解决办法

- 每遍历完一条边，更新完相应的边界后，都加一条判断 if (top > bottom || left > right) break，避免遍历完成时没有及时退出，导致重复遍历。
- 但你发现，遍历完成要么发生在遍历完“上边”，要么发生在遍历完“右边”
- 所以只需在这两步操作之后，加 if (top > bottom || left > right) break 即可

``` go
func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 {
		return []int{}
	}
	res := []int{}
	top, bottom, left, right := 0, len(matrix)-1, 0, len(matrix[0])-1
	for top <= bottom && left <= right {
		for i := left; i <= right; i++ { res = append(res, matrix[top][i]) }
		top++ 
		for i := top; i <= bottom; i++ { res = append(res, matrix[i][right]) }
		right--
		if top > bottom || left > right { break}
		for i := right; i >= left; i-- { res = append(res, matrix[bottom][i]) }
        bottom--
		for i := bottom; i >= top; i-- { res = append(res, matrix[i][left]) }
        left++
	}
	return res
}
```

### 换一种循环的条件，也是可以的
- 遍历完所有项时，res 数组构建完毕。我们可以用 res 数组的长度 等于 矩阵的项的个数，作为循环的结束条件
- 不等于就继续遍历，等于就 break

``` go
func spiralOrder(matrix [][]int) []int {
    if len(matrix) == 0 {
        return []int{}
    }
    res := []int{}
    top, bottom, left, right := 0, len(matrix)-1, 0, len(matrix[0])-1
    size := len(matrix) * len(matrix[0])
    
    for len(res) != size {
        for i := left; i <= right; i++ { res = append(res, matrix[top][i]) }
        top++
        for i := top; i <= bottom; i++ { res = append(res, matrix[i][right]) }
        right--
        if len(res) == size { break } 
        for i := right; i >= left; i-- { res = append(res, matrix[bottom][i]) }
        bottom--
        for i := bottom; i >= top; i-- { res = append(res, matrix[i][left]) }
        left++ 
    }

    return res
}

```










[59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)

构建 n * n 的矩阵

确定矩阵的四个边界，它是初始遍历的边界。

按 上 右 下 左，一层层向内，遍历矩阵填格子

每遍历一个格子，填上对应的 num，num 自增

直到 num > n*n ，遍历结束



``` go
func generateMatrix(n int) [][]int {
	matrix, num := make([][]int, n), 1
	for i := range matrix {
		matrix[i] = make([]int, n)
	}
	left, right, top, bottom := 0, n-1, 0, n-1
	for num <= n*n {
		for i := left; i <= right; i++ {
			matrix[top][i] = num
			num++
		}
		top++
		for i := top; i <= bottom; i++ {
			matrix[i][right] = num
			num++
		}
		right--
		for i := right; i >= left; i-- {
			matrix[bottom][i] = num
			num++
		}
		bottom--
		for i := bottom; i >= top; i-- {
			matrix[i][left] = num
			num++
		}
		left++
	}
	return matrix
}

```

复杂度分析

- 时间复杂度：O(n^2)
- 空间复杂度：O(n^2)