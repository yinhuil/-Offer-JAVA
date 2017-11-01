package JianZhiOffer;

public class Interview8MinNumber {
//	本题是寻找旋转数组的最小数字，    所以这个数字具备左边的比它大，右边的比他大的特征。   
	public static int MinNumber(int[] in){
		int len=in.length;
		int start=0;
		int end=len-1;
		int mid=start;
		while(in[start]>=in[end]){
			if((end-start)==1){
				mid=end;
				break;
			}
			mid=start+(end-start)/2;
			if(in[start]==in[mid] && in[mid]==in[end]){
				return minInorder(in,start,end);
			}
			if(in[mid]>=in[start]){
				start=mid;
			}else if(in[mid]<=in[end]){
				end=mid;
			}
		}
		return in[mid];	
	}
	private static int minInorder(int[] in, int start, int end) {
		int temp=in[start];
		for(int i=start+1;i<=end;i++){
			if(in[i]<temp){
				temp=in[i];
			}
		}
		return temp;
	}
	public static void main(String[] args) {
		int[] in={3,4,5,1,2};
		System.out.println(MinNumber(in));
	}
}
