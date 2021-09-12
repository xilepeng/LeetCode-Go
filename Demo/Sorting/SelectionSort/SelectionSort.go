package main

import (
	"fmt"
)

func selection_sort(A []int, n int) {
	for i := 0; i < x; i++ { //x -> n-1 (x后面已完成)
		min := i
		for j := i + 1; j < n; j++ {
			if A[j] < A[min] {
				min = j
			}
		}
		A[i], A[min] = A[min], A[i]
	}
}

var n, x int

func main() {
	fmt.Scanf("%d %d", &n, &x)
	A := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
	}
	selection_sort(A, n)
	for i := 0; i < n; i++ {
		fmt.Printf("%d ", A[i])
	}
}
