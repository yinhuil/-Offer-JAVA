package JianZhiOffer;

public class multiply {
/*	题目描述
	给定一个数组A[0,1,...,n-1],请构建一个数组
	B[0,1,...,n-1],其中B中的元素
	B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。*/
	
/*思路： 先负责前面的相乘，然后反过来进行后面的相乘；
 * */
	
	public int[] multiply(int[] A) {
		int length=A.length;
		int[] b=new int[length];
		if(length==0) return b;
		b[0]=1;
		for(int i=1;i<length;i++){
			b[i]=b[i-1]*A[i-1];
		}
		int temp=1;
		for(int j=length-2;j>=0;j--){
			temp*=A[j+1];
			b[j]*=temp;
		}
		return b;
	}
}
