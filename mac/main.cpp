#include <iostream>
#include <unistd.h>

void MyApplication();

int main()
{
	if (chdir(TIPS_DIR) != 0) {
		std::cout << "Could not change to " << TIPS_DIR << std::endl;
		return 1;
	}
	MyApplication();
	return 0;
}
