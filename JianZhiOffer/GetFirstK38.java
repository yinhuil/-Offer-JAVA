package JianZhiOffer;

public class GetFirstK38 {
	
/*	题目描述
	统计一个数字在排序数组中出现的次数。*/
	
//分别获取出现k的第一个位置和最后一个位置，   在获取位置的过程中使用二分法  
	public static int getFirstK(int[] arr,int k,int left,int right){
		if(left>right){
			return -1;
		}
		int middleIndex=(left+right)/2;
		int middleData=arr[middleIndex];
		if(middleData==k){
			if((middleIndex>0 && arr[middleIndex-1]!=k) || middleIndex==0){
				return middleIndex;
			}else{
				right=middleIndex-1;
			}
		}else if(middleData>k){
			right=middleIndex-1;
		}else{
			left=middleIndex+1;
		}
		return getFirstK(arr, k, left, right);
	}
	public static int getLastK(int[] arr,int k,int left,int right){
		if(left>right){
			return -1;
		}
		int middleIndex=(right+left)/2;
		int middleData=arr[middleIndex];
		if(middleData==k){
			if((middleIndex<right && arr[middleIndex+1]>k )|| middleIndex==right){
				return middleIndex;
			}else{
				left=middleIndex+1;
			}
		}else if(middleData>k){
			right=middleIndex-1;
		}else{
			left=middleIndex+1;
		}
		return getLastK(arr, k, left, right);
	}
	public static int getNumberOfK(int[] arr,int k){
		int number=0;
		if(arr.length>0){
			int first=getFirstK(arr, k, 0, arr.length-1);
			int last=getLastK(arr, k, 0, arr.length-1);
			if(first>-1 && last>-1){
				number=last-first+1;
			}
		}
		return number;
	}
	public static void main(String[] args) {
		int[] arr={1,2,3,3,3,3,4,5};
		System.out.println(getNumberOfK(arr, 3));

	}

}
