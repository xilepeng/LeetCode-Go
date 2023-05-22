
1. [155. 最小栈](#155-最小栈)
2. [34. 在排序数组中查找元素的第一个和最后一个位置](#34-在排序数组中查找元素的第一个和最后一个位置)
3. [补充题5. 手撕归并排序](#补充题5-手撕归并排序)
4. [154. 寻找旋转排序数组中的最小值 II](#154-寻找旋转排序数组中的最小值-ii)


## [155. 最小栈](https://leetcode.cn/problems/min-stack/)


```go
type MinStack struct {
	stack    []int
	minStack []int
}

/** initialize your data structure here. */
func Constructor() MinStack {
	return MinStack{
		stack:    []int{},
		minStack: []int{math.MaxInt64}, // 单调栈：单调递减，top 存储最小值
	}
}

func (this *MinStack) Push(x int) {
	this.stack = append(this.stack, x)
	top := this.minStack[len(this.minStack)-1]
	this.minStack = append(this.minStack, min(top, x))
}

func (this *MinStack) Pop() {
	this.stack = this.stack[:len(this.stack)-1]
	this.minStack = this.minStack[:len(this.minStack)-1]
}

func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}

func (this *MinStack) GetMin() int {
	return this.minStack[len(this.minStack)-1]
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

/**
 * Your MinStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(val);
 * obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.GetMin();
 */
```

```go
type MinStack struct {
	stack    []int
	minStack []int // 单调栈：单调递减，top 存储最小值
}

/** initialize your data structure here. */
func Constructor() MinStack {
	return MinStack{
		nil,
		nil,
	}
}

func (this *MinStack) Push(x int) {
	this.stack = append(this.stack, x)
	if len(this.minStack) == 0 || this.minStack[len(this.minStack)-1] >= x {
		this.minStack = append(this.minStack, x)
	}
}

func (this *MinStack) Pop() {
	if this.minStack[len(this.minStack)-1] == this.stack[len(this.stack)-1] {
		this.minStack = this.minStack[:len(this.minStack)-1]
	}
	this.stack = this.stack[:len(this.stack)-1]
}

func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}

func (this *MinStack) GetMin() int {
	return this.minStack[len(this.minStack)-1]
}

/**
 * Your MinStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.GetMin();
 */
```




## [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)



```go
func searchRange(nums []int, target int) []int {
	start, end := findFirst(nums, target), findLast(nums, target)
	return []int{start, end}
}
// 二分查找第一个与 target 相等的元素，时间复杂度 O(logn)
func findFirst(nums []int, target int) int {
	low, high := 0, len(nums)-1
	for low <= high {
		mid := low + (high-low)>>1
		if nums[mid] < target {
			low = mid + 1
		} else if nums[mid] > target {
			high = mid - 1
		} else {
			if mid == 0 || nums[mid-1] != target { // 找到第一个与 target 相等的元素
				return mid
			}
			high = mid - 1
		}
	}
	return -1
}
// 二分查找最后一个与 target 相等的元素，时间复杂度 O(logn)
func findLast(nums []int, target int) int {
	low, high := 0, len(nums)-1
	for low <= high {
		mid := low + (high-low)>>1
		if nums[mid] < target {
			low = mid + 1
		} else if nums[mid] > target {
			high = mid - 1
		} else {
			if mid == len(nums)-1 || nums[mid+1] != target { // 找到最后一个与 target 相等的元素
				return mid
			}
			low = mid + 1
		}
	}
	return -1
}
```






```go
func searchRange(nums []int, target int) []int {
	start, end := findFirst(nums, target), findLast(nums, target)
	return []int{start, end}
}

func findFirst(nums []int, target int) int {
	low, high, start := 0, len(nums)-1, -1
	for low <= high {
		mid := low + (high-low)>>1
		if nums[mid] >= target {
			high = mid - 1
		} else {
			low = mid + 1
		}
		if nums[mid] == target {
			start = mid
		}
	}
	return start
}

func findLast(nums []int, target int) int {
	low, high, end := 0, len(nums)-1, -1
	for low <= high {
		mid := low + (high-low)>>1
		if nums[mid] <= target {
			low = mid + 1
		} else {
			high = mid - 1
		}
		if nums[mid] == target {
			end = mid
		}
	}
	return end
}
```


```go

func searchRange(nums []int, target int) []int {
	return []int{searchFirstEqualElement(nums, target), searchLastEqualElement(nums, target)}

}

// 二分查找第一个与 target 相等的元素，时间复杂度 O(logn)
func searchFirstEqualElement(nums []int, target int) int {
	low, high := 0, len(nums)-1
	for low <= high {
		mid := low + ((high - low) >> 1)
		if nums[mid] > target {
			high = mid - 1
		} else if nums[mid] < target {
			low = mid + 1
		} else {
			if (mid == 0) || (nums[mid-1] != target) { // 找到第一个与 target 相等的元素
				return mid
			}
			high = mid - 1
		}
	}
	return -1
}

// 二分查找最后一个与 target 相等的元素，时间复杂度 O(logn)
func searchLastEqualElement(nums []int, target int) int {
	low, high := 0, len(nums)-1
	for low <= high {
		mid := low + ((high - low) >> 1)
		if nums[mid] > target {
			high = mid - 1
		} else if nums[mid] < target {
			low = mid + 1
		} else {
			if (mid == len(nums)-1) || (nums[mid+1] != target) { // 找到最后一个与 target 相等的元素
				return mid
			}
			low = mid + 1
		}
	}
	return -1
}

// 二分查找第一个大于等于 target 的元素，时间复杂度 O(logn)
func searchFirstGreaterElement(nums []int, target int) int {
	low, high := 0, len(nums)-1
	for low <= high {
		mid := low + ((high - low) >> 1)
		if nums[mid] >= target {
			if (mid == 0) || (nums[mid-1] < target) { // 找到第一个大于等于 target 的元素
				return mid
			}
			high = mid - 1
		} else {
			low = mid + 1
		}
	}
	return -1
}

// 二分查找最后一个小于等于 target 的元素，时间复杂度 O(logn)
func searchLastLessElement(nums []int, target int) int {
	low, high := 0, len(nums)-1
	for low <= high {
		mid := low + ((high - low) >> 1)
		if nums[mid] <= target {
			if (mid == len(nums)-1) || (nums[mid+1] > target) { // 找到最后一个小于等于 target 的元素
				return mid
			}
			low = mid + 1
		} else {
			high = mid - 1
		}
	}
	return -1
}
```


## [补充题5. 手撕归并排序](https://leetcode.cn/problems/sort-an-array/)

```go
func sortArray(nums []int) []int {
	arrLen := len(nums)
	if arrLen <= 1 {
		return nums
	}
	mergeSort(nums, 0, arrLen-1)
	return nums
}

func mergeSort(arr []int, start, end int) {
	if start >= end {
		return
	}
	mid := start + (end-start)>>1
	mergeSort(arr, start, mid)
	mergeSort(arr, mid+1, end)
	// merge(arr, start, mid, end)
	// merge1(arr, start, mid, end)
	merge2(arr, start, mid, end)
}

func merge(arr []int, start, mid, end int) {
	tmpArr := make([]int, end-start+1)
	i, j, k := start, mid+1, 0
	for ; i <= mid && j <= end; k++ {
		if arr[i] < arr[j] {
			tmpArr[k] = arr[i]
			i++
		} else {
			tmpArr[k] = arr[j]
			j++
		}
	}
	for ; i <= mid; i++ {
		tmpArr[k] = arr[i]
		k++
	}
	for ; j <= end; j++ {
		tmpArr[k] = arr[j]
		k++
	}
	copy(arr[start:end+1], tmpArr)
}

func merge1(arr []int, start, mid, end int) {
	tmpArr := make([]int, end-start+1)
	i, j, k := start, mid+1, 0
	for p := start; p <= end; p++ {
		if i > mid {
			tmpArr[k] = arr[j]
			j++
		} else if j > end {
			tmpArr[k] = arr[i]
			i++
		} else if arr[i] < arr[j] {
			tmpArr[k] = arr[i]
			i++
		} else {
			tmpArr[k] = arr[j]
			j++
		}
		k++
	}
	for p := 0; p < k; p++ {
		arr[start] = tmpArr[p]
		start++
	}
}

func merge2(arr []int, start, mid, end int) {
	tmpArr := make([]int, end-start+1)
	i, j, k := start, mid+1, 0
	for i <= mid || j <= end {
		if j > end || (i <= mid && arr[i] < arr[j]) {
			tmpArr[k] = arr[i]
			i++
		} else {
			tmpArr[k] = arr[j]
			j++
		}
		k++
	}
	copy(arr[start:end+1], tmpArr)
}
```




## [154. 寻找旋转排序数组中的最小值 II](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/description/)

```go
func findMin(nums []int) int {
	low, high := 0, len(nums)-1
	for low < high {
		mid := low + ((high - low) >> 1) // mid = (low + high)/2
		if nums[mid] < nums[high] {      // mid 在右排序区，旋转点在[low, mid]
			high = mid
		} else if nums[mid] > nums[high] { // mid 在左排序区，旋转点在[mid+1, high]
			low = mid + 1
		} else { // 无法判断 mid 在哪个排序数组中
			high--
		}
	}
	return nums[low]
}
```