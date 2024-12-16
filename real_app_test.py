import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from safetensors.torch import load_file

# Load trained model and tokenizer
model_dir = "./spam_detect_final"  # Set the correct path to your model directory
config_path = os.path.join(model_dir, "config.json")
model_path = os.path.join(model_dir, "model.safetensors")

# Load tokenizer from bert-base-chinese
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# Load configuration and model weights
config = BertConfig.from_json_file(config_path)
state_dict = load_file(model_path)

# Initialize model with loaded configuration and weights
model = BertForSequenceClassification(config)
model.load_state_dict(state_dict)

# Set model to evaluation mode
model.eval()

# Define a function for single-string spam detection
def predict_spam(message):
    """
    Predict whether the input message is spam or not.

    Args:
        message (str): The input message as a string.

    Returns:
        str: "Spam" or "Ham" (not spam).
    """
    # Tokenize input
    inputs = tokenizer(
        message,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        print(logits)
        #use softmax
        logits = torch.nn.functional.softmax(logits, dim=1)
        print(logits)
        # Get the predicted class
        spam_prob = logits[0][1].item()

    # Map prediction to label
    return "Spam" if spam_prob > 0.6 else "Ham"
    #return "Spam" if spam_prob == 1 else "Ham"

# Test the function
if __name__ == "__main__":
    # Example spam and ham messages
    test_messages = [
        '''@所有人 【诚信考试专项教育】直播已结束，感谢大家认真观看学习
            未完成观看或观看时长低于20分钟的同学，请于本周日前完成直播回放观看噢。
            期末考试即将来临，在此提醒以下几点：
            第一，千万不要作弊！千万不要作弊！千万不要作弊！
            第二，明确自己有哪些考试、明确考试时间、明确考场，不要漏考、迟到、跑错考场。
            第三，调整心态，不要过度焦虑，不论是考试还是申研，祝大家都幸运！
            天气变化，注意保暖，注意饮食健康，多喝温水，保重身体~''',
            "您好！贵公司经理/财务：\n      首先，对我的冒昧来函向您致歉，但愿这函对贵公司有所帮助。\n    我司享有国家优惠政策；纳税率低于一般纳税人，如贵司有下列\n    情况：\n    1，公司做进帐，出项有差额。\n    2，客户压底货价，利润微薄。\n    3，采购时需要正式票据报销。\n    我司长期为贵司提供如下票务项目（先用票后付款）\n     1电脑版增值税专用发票，国税（商品销售发票），地税（运输票，\n      广告票，服务票，建安票等）\n     2海关专用缴款书，其他票据\n     注：\n     1我司郑重承诺所有票据税务局代开或企业直接开出，并可以上网\n      查询。\n     2以上票据税率优惠，具体依票面金额大小而不同，欢迎您来电话\n     \t洽谈。\n      \n                详情请电:0755-21011747     \n                联 系 人:梁先生      \n\n              \n\n",
            "各位同学好，期末考试就要到了，请大家认真复习，争取取得好成绩。",
        '''【形势与政策II-专题教育】报名参加通知---（通知面向ZJE和ZJUI二年级、三年级、四年级本科生，同学们请扫描以下的二维码报名噢，不要扫推文中的二维码报名）
            嗨同学们！2024年12月10日14:00将举办一场主题为“Marco Polo: a Model for Businessmen, Novelists and Ambassadors-Sino-ltalian Cultural Exchanges as Viewed by Marco Polo”的报告，此报告将作为形势与政策II的一讲，具体信息如下：

            开始报名时间：2024年12月9日16:30
            Registration begins at: 16:30, December 9, 2024

            主题：“Marco Polo: a Model for Businessmen, Novelists and Ambassadors-Sino-ltalian Cultural Exchanges as Viewed by Marco Polo”

            报告时间:  2024年12月10日 14:00 （北京时间）
            Time: December 10th 14:00

            地点：4号书院多动能厅
            Location：Multifunctional Hall NO. 4 Residential College

            语言:英语
            Language:English

            主讲嘉宾：Pier Francesco FUMAGALLI, 中文名傅马利
            南开大学区域国别研究中心客座研究员，2024 - 2029
            米兰昂布罗修图书馆名誉博士，2023 - 至今
            浙江传媒学院客座教授，2024-2027
            西安西北大学客座教授，2023-2026
            米兰意中友协董事会成员，2000- 至今     
            米兰昂布罗修图书馆博士，1981-2023
            米兰昂布罗修图书馆东方部主任，2008-2023
            米兰昂布罗修图书馆副馆长，2007-2017
            浙江大学兼职教授，2008-2014
            米兰天主教大学合同教授，2006-2021

            注意：
            本次报告会作为‘形势与政策II’课程的一讲，面向ZJE和ZJUI二年级、三年级、四年级
            本科生开放，共50个名额。请扫描以下二维码完成报名，信息提交成功即为报名成功。名额满后，系统将自动关闭。''',
            "@所有人 同学们，我们邀请到昆山杜克大学协理副校长、杜克大学教授李昕博士拟于下周四（12.19）17点30-19点30在校区开展讲座（讲座地点待定），内容包括：1、从招生委员会主任视角解读美国名校硕士申请；2、杜克大学单学位硕士介绍（ECE、环境政策、管理学、全球健康、医学物理5个项目），为方便安排讲座场地，特邀请感兴趣的同学填写此预报名问卷，并提前收集大家想要咨询的相关问题【https://forms.office.com/r/ZsWCv7WDCm】，谢谢大家！",
            "尊敬的客户，您的信用卡积分即将过期，点击此处兑换豪礼！",
            "全场五折优惠，限时抢购！快来我们的官网选购吧！",
            '''上海、北京、广州、香港和台北出发的航班现已推出优惠票
            价，这是探索新城市的绝佳时机！立即访问土耳其航空官网
            或移动应用程序，享受专属优惠，成为这场非凡旅行体验的
            一部分。2024年11月18日''',
            '''各位师生：
            冬至至日日初长，久客客怀怀故乡。在冬至即将到来之际，食堂节气食育活动又开始啦！
            “面团圆圆杖下扁，筷子取馅面中填。巧手捏出玲珑褶，饺子浮沉几人馋？”让我们一起在冬至的节气中，享受这一刻带来的温馨与满足。
            为了更好地服务广大师生，提高食堂的服务水平，食堂推出节气食育活动之饺子制作，邀请大家一起参与，亲手包制饺子。
            1.	活动对象
            国际校区师生（报名参加，限额25人）
            2.	活动时间
            2024年12月19日（周四）14:00-15:00
            3.	活动地点
            国际校区食堂一楼西餐厅
            4.	活动内容
            现场提前准备了饺子制作的材料，由食堂师傅指导进行饺子制作，一共制作三种口味（猪肉白菜、鲜肉玉米、酸菜油渣）。活动后可将亲手制作的饺子带走与家人或朋友分享，在快乐的时刻里，让美味成为永恒的香气。
            5.	活动报名
            请于12月17日11：00前填写下方报名表，报名成功后会收到邮件通知，名额有限，先到先得。
            ''',
    ]

    # Predict and display results
    for msg in test_messages:
        result = predict_spam(msg)
        print(f"Message: {msg}\nPrediction: {result}\n")
