import os
root = os.getcwd()
os.chdir("./data_collect/cat")
l = os.listdir()
for i in range(len(l)):
    os.rename(l[i], "cat_"+str(i))
    try:
        os.rename(root+"/대표이미지/"+l[i]+".jpg", root+"/대표이미지/cat_"+str(i)+".jpg")
    except:
        os.rename(root+"/대표이미지/"+l[i]+".jpeg", root+"/대표이미지/cat_"+str(i)+".jpeg")

    print(f"rename {l[i]} to cat {i}")

"""
rename insta_14 to cat 0
rename insta_15 to cat 1
rename insta_16 to cat 2
rename insta_17 to cat 3
rename insta_18 to cat 4
rename insta_19 to cat 5
rename insta_2 to cat 6
rename insta_20 to cat 7
rename insta_21 to cat 8
rename insta_22 to cat 9
rename insta_23 to cat 10
rename insta_24 to cat 11
rename insta_25 to cat 12
rename insta_26 to cat 13
rename insta_27 to cat 14
rename insta_28 to cat 15
rename insta_29 to cat 16
rename insta_3 to cat 17
rename insta_30 to cat 18
rename insta_31 to cat 19
rename insta_32 to cat 20
rename insta_33 to cat 21
rename insta_34 to cat 22
rename insta_35 to cat 23
rename insta_36 to cat 24
rename insta_37 to cat 25
rename insta_38 to cat 26
rename insta_39 to cat 27
rename insta_4 to cat 28
rename insta_40 to cat 29
rename insta_41 to cat 30
rename insta_42 to cat 31
rename insta_43 to cat 32
rename insta_44 to cat 33
rename insta_45 to cat 34
rename insta_46 to cat 35
rename insta_47 to cat 36
rename insta_48 to cat 37
rename insta_49 to cat 38
rename insta_5 to cat 39
rename insta_50 to cat 40
rename insta_51 to cat 41
rename insta_52 to cat 42
rename insta_53 to cat 43
rename insta_54 to cat 44
rename insta_55 to cat 45
rename insta_56 to cat 46
rename insta_57 to cat 47
rename insta_58 to cat 48
rename insta_59 to cat 49
rename insta_6 to cat 50
rename insta_60 to cat 51
rename insta_62 to cat 52
rename insta_63 to cat 53
rename insta_65 to cat 54
rename insta_66 to cat 55
rename insta_67 to cat 56
rename insta_68 to cat 57
rename insta_69 to cat 58
rename insta_7 to cat 59
rename insta_70 to cat 60
rename insta_71 to cat 61
rename insta_72 to cat 62
rename insta_73 to cat 63
rename insta_74 to cat 64
rename insta_75 to cat 65
rename insta_76 to cat 66
rename insta_77 to cat 67
rename insta_78 to cat 68
rename insta_79 to cat 69
rename insta_8 to cat 70
rename insta_80 to cat 71
rename insta_81 to cat 72
rename insta_82 to cat 73
rename insta_83 to cat 74
rename insta_84 to cat 75
rename insta_85 to cat 76
rename insta_87 to cat 77
rename insta_88 to cat 78
rename insta_89 to cat 79
rename insta_9 to cat 80
rename insta_90 to cat 81
rename 라떼 to cat 82
rename 마끼 to cat 83
rename cat_0 to cat 84
rename insta_1 to cat 85
rename insta_10 to cat 86
rename insta_11 to cat 87
"""