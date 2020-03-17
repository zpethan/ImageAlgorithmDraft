#include "DataStructAndAlgorithm.h"

int prime(int p)
{
	if (p <= 1){ return 0; }
	if (p == 2 || p == 3){ return 1; }
	if (p % 6 != 1 && p % 6 != 5){ return 0; }
	for (int i = 5; i <= floor(sqrt((float)p)); i += 6)
	{
		if (p%i == 0 || p % (i + 2) == 0){ return 0; }
	}
	return 1;
}

