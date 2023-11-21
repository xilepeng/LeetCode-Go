
1. [46. 全排列](#46-全排列)
2. [47. 全排列 II  补充](#47-全排列-ii--补充)
3. [155. 最小栈](#155-最小栈)
4. [34. 在排序数组中查找元素的第一个和最后一个位置](#34-在排序数组中查找元素的第一个和最后一个位置)
5. [补充题5. 手撕归并排序](#补充题5-手撕归并排序)
6. [154. 寻找旋转排序数组中的最小值 II](#154-寻找旋转排序数组中的最小值-ii)
7. [400. 第 N 位数字](#400-第-n-位数字)





## [46. 全排列](https://leetcode-cn.com/problems/permutations/)

**枚举每个位置，填每个数 (回溯)**

```go
func permute(nums []int) [][]int {
	res, path, used, n := [][]int{}, []int{}, make([]bool, len(nums)), len(nums)
	var dfs func(int)
	
	dfs = func(pos int) {   // 枚举位置
		if len(path) == n { // pos == n 
			res = append(res, append([]int{}, path...)) // path append后会扩容，消除前面的无效数据(0)
			return
		}
		for i := 0; i < n; i++ { // 枚举所有的选择
			if !used[i] {        // 第i个位置未使用
				used[i] = true               // 第i个位置已使用
				path = append(path, nums[i]) // 做出选择，记录路径
				dfs(pos + 1)                 // 枚举下一个位置
				used[i] = false              // 撤销选择
				path = path[:len(path)-1]    // 取消记录
			}
		}
	}

	dfs(0)
	return res
}
```

```go
func permute(nums []int) [][]int {
	res, n := [][]int{}, len(nums)
	var dfs func(int)

	dfs = func(pos int) {
		if pos == n { // 所有位置都已填满
			res = append(res, append([]int{}, nums...))
			return // 结束递归
		}
		for i := pos; i < len(nums); i++ {
			nums[pos], nums[i] = nums[i], nums[pos] // pos 位置填入 nums[i]
			dfs(pos + 1)                            // 递归填下一个位置
			nums[pos], nums[i] = nums[i], nums[pos] //撤销、回溯
		}
	}

	dfs(0)
	return res
}
```


## [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)  补充

```go
func permuteUnique(nums []int) [][]int {
	sort.Ints(nums)
	used, res, path := make([]bool, len(nums)), [][]int{}, []int{}
	var dfs func(int)

	dfs = func(pos int) {
		if len(path) == len(nums) {
			res = append(res, append([]int{}, path...))
			return
		}
		for i := 0; i < len(nums); i++ {
			if used[i] || i > 0 && !used[i-1] && nums[i-1] == nums[i] { // 已使用 或 重复
				continue // 去重，跳过
			}
			used[i] = true
			path = append(path, nums[i])
			dfs(pos + 1)
			used[i] = false
			path = path[:len(path)-1]
		}
	}

	dfs(0)
	return res
}
```



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


## [400. 第 N 位数字](https://leetcode.cn/problems/nth-digit/description/)

```go

func findNthDigit(n int) int {
	start, digit := 1, 1
	for n > 9*start*digit {
		n -= 9 * start * digit
		start *= 10
		digit++
	}
	num := start + (n-1)/digit
	digitIndex := (n - 1) % digit
	return int(strconv.Itoa(num)[digitIndex] - '0')
}

func findNthDigit1(n int) int {
	digit, start, count := 1, 1, 9
	for n > count {
		n -= count
		start *= 10
		digit++
		count = 9 * start * digit
	}
	num := start + (n-1)/digit
	index := (n - 1) % digit
	return int((strconv.Itoa(num)[index]) - '0')
}

func findNthDigit2(n int) int {
	if n <= 9 {
		return n
	}
	bits := 1
	for n > 9*int(math.Pow10(bits-1))*bits {
		n -= 9 * int(math.Pow10(bits-1)) * bits
		bits++
	}
	index := n - 1
	start := int(math.Pow10(bits - 1))
	num := start + index/bits
	digitIndex := index % bits
	return num / int(math.Pow10(bits-digitIndex-1)) % 10
}
```

