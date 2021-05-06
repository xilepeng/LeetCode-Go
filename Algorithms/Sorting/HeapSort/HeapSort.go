package main

import (
	"fmt"
)

func heap_sort(A []int) {
	heap_size := len(A)
	build_maxheap(A, heap_size)
	for i := heap_size - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]
		heap_size--
		max_heapify(A, 0, heap_size)
	}
}
func build_maxheap(A []int, heap_size int) {
	for i := heap_size >> 1; i >= 0; i-- {
		max_heapify(A, i, heap_size)
	}
}
func max_heapify(A []int, i, heap_size int) {
	l, r, largest := i<<1+1, i<<1+2, i
	for l < heap_size && A[largest] < A[l] {
		largest = l
	}
	for r < heap_size && A[largest] < A[r] {
		largest = r
	}
	if largest != i {
		A[largest], A[i] = A[i], A[largest]
		max_heapify(A, largest, heap_size)
	}
}

func main() {
	var n int
	fmt.Scanf("%d", &n)
	A := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
	}
	heap_sort(A)
	fmt.Println(A)
}
