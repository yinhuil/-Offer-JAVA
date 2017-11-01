package JianZhiOffer;
/*题目描述
求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为
此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,
但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很
快的求出任意非负整数区间中1出现的次数*/
public class NumberOf1Between1AndN32 {
/*方法一，从个位开始数1的个数，记下这个值为current，前面的值为before，后面的值为after，  举个例子 156，当十位大于1的时候 也就是10-19,110-119，一共有（before+1）*10个
	1，如果116这个值的时候，也就是10-19 110-116一共有before*10+after+1， 如果是106的时候，也就是10-19，一共也就是before*10个数字，这样依次从
	个位到十位到百位等等*/
    public static int NumberOf1Between1AndN_Solution1(int n) {
    	int count=0;
    	int i=1;
    	int current=0,after=0,before=0;
    	while((n/i)!=0){
    		current=(n/i)%10;
    		before=n/(i*10);
    		after=n-(n/i)*i;
    		if(current==0){
    			count+=before*i;
    		}else if(current==1){
    			count+=before*i+after+1;
    		}else{
    			count+=(before+1)*i;
    		}
    		i=i*10;
    	}
    	return count;
    }
	// 时间复杂度为O（logn）
    
    
/* 
方法二中，当数值很大的时候，我们用字符串的形式进行保存，假如是12345这个数值，我们先从最高位进行算起，也就是10000-19999，这样最高位一共是10000个
但是还有2346-12345中其余0的位数， first * (length - 1) * PowerBase10(length - 2); 
*/
	public static int NumberOf1Between1AndN_Solution(int n) {

		if (n <= 0) {
			return 0;
		}

		StringBuilder s = new StringBuilder(((Integer) n).toString());

		return NumberOf1(s);

	}

	static int NumberOf1(StringBuilder s) {
		if (s == null || s.length() == 0 || s.charAt(0) < '0'
				|| s.charAt(s.length() - 1) > '9') {
			return 0;
		}

		int first = s.charAt(0) - '0';

		int length = s.length();

		if (length == 1 && first == 0) {
			return 0;
		}

		if (length == 1 && first > 0) {
			return 1;
		}

		// 假设n = 21345
		// numFirstDigit是数字10000 - 19999 的第一个位中的数目
		int numFirstDigit = 0;
		if (first > 1) {
			numFirstDigit = PowerBase10(length - 1);

		} else if (first == 1) {
			numFirstDigit = Integer.parseInt(s.substring(1)) + 1;

		}
		// numOtherDigits是1346 - 21345除了第一位之外的数位中的数目
		int numOtherDigits = first * (length - 1) * PowerBase10(length - 2);
		// numRecursive是1 - 1345中的数目
		int numRecursive = NumberOf1(s.deleteCharAt(0));

		return numFirstDigit + numOtherDigits + numRecursive;
	}

	static int PowerBase10(int n) {
		int result = 1;
		for (int i = 0; i < n; i++) {
			result *= 10;

		}

		return result;
	}
	
	public static void main(String[] args) {
		System.out.println(NumberOf1Between1AndN_Solution(30));
	}
}
