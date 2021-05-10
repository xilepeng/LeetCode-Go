


[167. 两数之和 II - 输入有序数组](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/)

[88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

[26. 删除有序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

[76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

[32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)

[155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

[42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

[84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

[239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

[918. 环形子数组的最大和](https://leetcode-cn.com/problems/maximum-sum-circular-subarray/)




[84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)


## 方法一：暴力

如果我们枚举「高」，我们可以使用一重循环枚举某一根柱子，将其固定为矩形的高度 h。随后我们从这跟柱子开始向两侧延伸，直到遇到高度小于 h 的柱子，就确定了矩形的左右边界。如果左右边界之间的宽度为 w，那么对应的面积为 w * h。

Time Limit Exceeded

```go
func largestRectangleArea(heights []int) (res int) {
	n := len(heights)
	for mid := 0; mid < n; mid++ {
		height, left, right := heights[mid], mid, mid
		for ; left-1 >= 0 && heights[left-1] >= height; left-- {
		}
		for ; right+1 < n && heights[right+1] >= height; right++ {
		}
		if res < (right-left+1)*height {
			res = (right - left + 1) * height
		}
	}
	return
}
```

单调栈：查找每个数左侧第一个比它小的数
单调队列：滑动窗口中的最值

## 方法一：单调栈

```go
func largestRectangleArea(heights []int) int {
	n := len(heights)
	left, right := make([]int, n), make([]int, n)
	mono_stack := []int{}
	for i := 0; i < n; i++ {
		for len(mono_stack) > 0 && heights[mono_stack[len(mono_stack)-1]] >= heights[i] {
			mono_stack = mono_stack[:len(mono_stack)-1]
		}
		if len(mono_stack) == 0 {
			left[i] = -1
		} else {
			left[i] = mono_stack[len(mono_stack)-1]
		}
		mono_stack = append(mono_stack, i)
	}
	mono_stack = []int{}
	for i := n - 1; i >= 0; i-- {
		for len(mono_stack) > 0 && heights[mono_stack[len(mono_stack)-1]] >= heights[i] {
			mono_stack = mono_stack[:len(mono_stack)-1]
		}
		if len(mono_stack) == 0 {
			right[i] = n
		} else {
			right[i] = mono_stack[len(mono_stack)-1]
		}
		mono_stack = append(mono_stack, i)
	}
	ans := 0
	for i := 0; i < n; i++ {
		ans = max(ans, (right[i]-left[i]-1)*heights[i])
	}
	return ans
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

复杂度分析

- 时间复杂度：O(N)。
- 空间复杂度：O(N)。
