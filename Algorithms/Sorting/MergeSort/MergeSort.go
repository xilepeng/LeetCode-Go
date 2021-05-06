package main

import (
	"fmt"
)

func merge_sort(A []int, start, end int) {
	if start < end {
		mid := (start + end) >> 1
		merge_sort(A, start, mid)
		merge_sort(A, mid+1, end)
		merge(A, start, mid, end)
	}
}
func merge(A []int, start, mid, end int) {
	tmp := []int{}
	i, j := start, mid+1
	for i <= mid || j <= end {
		if i > mid || j <= end && A[j] < A[i] {
			tmp = append(tmp, A[j])
			j++
		} else {
			tmp = append(tmp, A[i])
			i++
		}
	}
	copy(A[start:end+1], tmp)
}

func main() {
	var n int
	fmt.Scanf("%d", &n)
	A := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
	}
	merge_sort(A, 0, len(A)-1)
	fmt.Println(A)
}
