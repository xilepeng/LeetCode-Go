package singleton

type singleton struct{}

type Instance interface {
	Work() string
}

var s *singleton

// 01_饿汉式单例模式
func init() {
	s = newSgingleton()
}

func newSgingleton() *singleton {
	return &singleton{}
}

func (s *singleton) Work() string {
	return "singleton is working ..."
}
func GetInstance() Instance {
	return s
}
