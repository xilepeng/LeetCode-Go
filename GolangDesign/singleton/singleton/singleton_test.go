package singleton

import "testing"

func TestSingleton(t *testing.T) {
	s := GetInstance()
	putInstance(s)
	t.Error(s.Work())
}

func putInstance(s Instance) {
}
