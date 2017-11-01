package JianZhiOffer;

public class Sum_Solution {
	
/*	题目描述
	求1+2+3+...+n，要求不能使用乘除法、
	for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。*/	
	
	
//解题思路，利用短路的思想，如果前面这个为false的时候也就是为0的时候，就可以停止运算了	
	public static int Sum_Solution(int n) {
		int sum = n;
        boolean ans = (n>0)&&((sum+=Sum_Solution(n-1))>0);
        return sum;
	}
	
	public static void main(String[] args) {

		int n=10;
		System.out.println(Sum_Solution(n));
	}
}
