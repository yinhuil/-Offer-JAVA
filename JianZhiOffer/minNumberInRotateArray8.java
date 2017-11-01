package JianZhiOffer;

public class minNumberInRotateArray8 {
	public static int minNumberInRotateArray(int[] array) {
		if(array.length==0) return 0;
		int start=0,end=array.length-1;
		int mid=0;
		while(array[start]>=array[end]){
			if(end-start==1){
				mid=end;
				break;
			}
			mid=start+(end-start)/2;
			if(array[start]==array[end] && array[start]==array[mid]){
				return minorder(array,start,end);
			}
			if(array[mid]>=array[start]){
				start=mid;
			}else{
				end=mid;
			}
		}
		return array[mid];
	}

	private static int minorder(int[] array, int start, int end) {
		int result=array[start];
		for(int i=start+1;i<=end;i++){
			if(array[i]<result){
				result=array[i];
			}
		}
		return result;
	}
	public static void main(String[] args) {
		int[] in={3,4,5,1,2};
		System.out.println(minNumberInRotateArray(in));
	}
}
