package JianZhiOffer;

//输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。


//思路： 当一个数字和他的减一的数字与的时候，就会相当于减少一个1的个数，这样的话，全程运算过程中就只运算了1的个数不用考虑0的个数，从而减少了运行复杂度
//假如（1110）&（1101）=1100；
public class NumberOf1 {
	public static int NumberOf1(int in) {
		int ans = 0;
		while (in != 0) {
			ans++;
			in = in & (in - 1);
		}
		return ans;
	}
	public static void main(String[] args) {
		System.out.println(NumberOf1(14));
	}
}
