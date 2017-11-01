package JianZhiOffer;

public class PowerWithUnsignedExponent {
	public static double PowerWithUnsignedExponent(double base, int exponent) {
		if (exponent == 0)
			return 1;
		if (exponent == 1)
			return base;
		double result = PowerWithUnsignedExponent(base, exponent >> 1);
		result *= result;
		if ((exponent&0x1)==1) {
			result *= base;
		}
		return result;
	}

	public static void main(String[] args) {
		
		System.out.println(5>>1);
	}
}
