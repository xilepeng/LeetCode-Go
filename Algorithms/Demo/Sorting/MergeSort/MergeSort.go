package main

import (
	"fmt"
	"io/ioutil"
	"os"
)

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

func main() {
	var n int
	_, _ = fmt.Scanln(&n)
	b, _ := ioutil.ReadAll(os.Stdin)
	A, Arr := make([]int, n), make([]int, n)
	num, i := 0, 0
	for _, by := range b {
		if by == ' ' {
			A[i] = num
			num = 0
			i++
		} else {
			num = num*10 + int(by-'0')
		}
	}
	A[i] = num

	fmt.Println(merge_sort(A, Arr, 0, n-1))
}
