## 《剑指 Offer》










[剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

------








[剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

### 方法一：Merge Sort

```go
func reversePairs(nums []int) int {
    n := len(nums)
    temp := make([]int, n)
    return merge_sort(nums, temp, 0, n-1)
}
func merge_sort(A, temp []int, start, end int) int {
    if start >= end {
        return 0
    }
    mid := start + (end - start)>>1
    count := merge_sort(A, temp, start, mid) + merge_sort(A, temp, mid+1, end)
    i, j, k := start, mid+1, 0
    for ; i <= mid && j <= end; k++ {
        if A[i] > A[j] {
            count += mid-i+1
            temp[k] = A[j]
            j++
        } else {
            temp[k] = A[i]
            i++
        }
    }
    for ; i <= mid; i++ {
        temp[k] = A[i]
        k++
    }
    for ; j <= end; j++ {
        temp[k] = A[j]
        k++
    }
    copy(A[start:end+1], temp)
    return count
}
```

### 方法二：Merge Sort

```go
func reversePairs(nums []int) int {
    n := len(nums)
    Arr := make([]int, n)
    return merge_sort(nums,Arr, 0, n-1)
}
func merge_sort(A, Arr []int, start, end int) int {
	if start >= end {
		return 0
	}
	mid := start + (end-start)>>1
	left := merge_sort(A, Arr, start, mid)
	right := merge_sort(A, Arr, mid+1, end)
	cross := merge(A, Arr, start, mid, end)
	return left + right + cross
}
func merge(A, Arr []int, start, mid, end int) int {
	p, q, k, count := start, mid+1, 0, 0
	for i := start; i <= end; i++ {
		if p > mid {
			Arr[k] = A[q]
			q++
		} else if q > end {
			Arr[k] = A[p]
			p++
		} else if A[p] <= A[q] {
			Arr[k] = A[p]
			p++
		} else {
			Arr[k] = A[q]
			q++
			count += mid - p + 1
		}
		k++
	}
	for p = 0; p < k; p++ {
		A[start] = Arr[p]
		start++
	}
	return count
}
```

### 方法三：Merge Sort

```go
func reversePairs(nums []int) int {
    n := len(nums)
    return merge_sort(nums, 0, n-1)
}
func merge_sort(A []int, start, end int) int {
	if start >= end {
		return 0
	}
	mid := (start + end) >> 1
	left := merge_sort(A, start, mid)
	right := merge_sort(A, mid+1, end)
	cross := merge(A, start, mid, end)
	return left + right + cross
}
func merge(A []int, start, mid, end int) int {
	temp, count := []int{}, 0
	i, j := start, mid+1
	for i <= mid && j <= end {
		if A[i] > A[j] {
			count += mid - i + 1
			temp = append(temp, A[j])
			j++
		} else {
			temp = append(temp, A[i])
			i++
		}
	}
	for ; i <= mid; i++ {
		temp = append(temp, A[i])
	}
	for ; j <= end; j++ {
		temp = append(temp, A[j])
	}
	copy(A[start:end+1], temp)
	return count
}
```

