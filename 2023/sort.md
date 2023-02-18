



## [补充题4. 手撕快速排序 (912. 排序数组)](https://leetcode-cn.com/problems/sort-an-array/)


**解法一**

```go
func sortArray(nums []int) []int {
	quickSort(nums, 0, len(nums)-1)
	return nums
}

func quickSort(A []int, start, end int) {
	if start >= end {
		return
	}
	x := A[(start+end)>>1]  // x := A[(start+end)/2],用j划分递归子区间
	i, j := start-1, end+1 // 循环内直接扫描下一个数，导致多操作1次，所以预处理
	for i < j {
		for i++; A[i] < x; i++ { // 从左向右扫描，找到大于 x 的数，停止
		}
		for j--; A[j] > x; j-- { // 从右向左扫描，找到小于 x 的数，停止
		}
		if i < j {
			A[i], A[j] = A[j], A[i] // 交换, 使得左边小于 x, 右边大于 x
		}
	}
	quickSort(A, start, j) // 递归处理 x 左边
	quickSort(A, j+1, end) // 递归处理 x 右边
}
```

**解法二**


```go
func sortArray(nums []int) []int {
    rand.Seed(time.Now().UnixNano())
    quick_sort(nums, 0, len(nums)-1)
    return nums
}

func quick_sort(A []int, start, end int) {
    if start < end {
        piv_pos := random_partition(A, start, end)
        quick_sort(A, start, piv_pos-1)
        quick_sort(A, piv_pos+1, end)
    }
}

func partition(A []int, start, end int) int {
    i, piv := start, A[end] // 从第一个数开始扫描，选取最后一位数字最为对比
    for j := start; j < end; j++ {
        if A[j] < piv {
            if i != j {// 不是同一个数
                A[i], A[j] = A[j], A[i]// A[j] 放在正确的位置
            }
            i++//扫描下一个数
        }
    }
    A[i], A[end] = A[end], A[i] // A[end] 回到正确的位置
    return i 
}

func random_partition(A []int, start, end int) int {
    random := rand.Int()%(end-start+1)+start
    A[random], A[end] = A[end],A[random]
    return partition(A, start, end)
}
```

## [补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)




```go
func sortArray(nums []int) []int {
	heapSort(nums)
	return nums
}

func heapSort(nums []int) {
	heapSize := len(nums)
	buildMaxHeap(nums, heapSize)
	for i := heapSize - 1; i >= 1; i-- {
		nums[0], nums[i] = nums[i], nums[0]
		heapSize--
		maxHeapify(nums, 0, heapSize)
	}
}

func buildMaxHeap(nums []int, heapSize int) {
	for i := heapSize >> 1; i >= 0; i-- {
		maxHeapify(nums, i, heapSize)
	}
}

func maxHeapify(nums []int, i, heapSize int) {
	for i<<1+1 < heapSize {
		lson, rson, large := i<<1+1, i<<1+2, i
		if lson < heapSize && nums[large] < nums[lson] {
			large = lson
		}
		for rson < heapSize && nums[large] < nums[rson] {
			large = rson
		}
		if large != i {
			nums[i], nums[large] = nums[large], nums[i]
			i = large
		} else {
			break
		}
	}
}

func maxHeapify1(nums []int, i, heapSize int) {
	lson, rson, large := i<<1+1, i<<1+2, i
	if lson < heapSize && nums[large] < nums[lson] {
		large = lson
	}
	if rson < heapSize && nums[large] < nums[rson] {
		large = rson
	}
	if large != i {
		nums[i], nums[large] = nums[large], nums[i]
		maxHeapify(nums, large, heapSize)
	}
}
```




```go

func bubbleSort(nums []int) {
	n := len(nums)
	for i := 0; i < n; i++ {
		flag := false
		for j := 0; j < n-i-1; j++ {
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
				flag = true
			}
		}
		if !flag {
			break
		}
	}
}
```