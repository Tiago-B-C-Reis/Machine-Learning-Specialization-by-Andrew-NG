import numpy as np

X = np.array([[1.84207953, 4.6075716],
               [5.65858312, 4.79996405],
               [6.35257892, 3.2908545],
               [2.90401653, 4.61220411],
               [3.23197916, 4.93989405],
               [1.24792268, 4.93267846],
               [1.97619886, 4.43489674],
               [2.23454135, 5.05547168],
               [2.98340757, 4.84046406],
               [2.97970391, 4.80671081],
               [2.11496411, 5.37373587],
               [2.12169543, 5.20854212],
               [1.5143529, 4.77003303],
               [2.16979227, 5.27435354],
               [0.41852373, 4.88312522],
               [2.47053695, 4.80418944],
               [4.06069132, 4.99503862],
               [3.00708934, 4.67897758],
               [0.66632346, 4.87187949],
               [3.1621865, 4.83658301],
               [0.51155258, 4.91052923],
               [3.1342801, 4.96178114],
               [2.04974595, 5.6241395],
               [0.66582785, 5.24399257],
               [1.01732013, 4.84473647],
               [2.17893568, 5.29758701],
               [2.85962615, 5.26041997],
               [1.30882588, 5.30158701],
               [0.99253246, 5.01567424],
               [1.40372638, 4.57527684],
               [2.66046572, 5.19623848],
               [2.79995882, 5.11526323],
               [2.06995345, 4.6846713],
               [3.29765181, 5.59205535],
               [1.8929766, 4.89043209],
               [2.55983064, 5.26397756],
               [1.15354031, 4.67866717],
               [2.25150754, 5.4450031],
               [2.20960296, 4.91469264],
               [1.59141937, 4.83212573],
               [1.67838038, 5.26903822],
               [2.59148642, 4.92593394],
               [2.80996442, 5.53849899],
               [0.95311627, 5.58037108],
               [1.51775276, 5.03836638],
               [3.23114248, 5.78429665],
               [2.54180011, 4.81098738],
               [3.81422865, 4.73526796],
               [1.68495829, 4.59643553],
               [2.17777173, 4.86154019],
               [1.8173328, 5.13333907],
               [1.85776553, 4.86962414],
               [3.03084301, 5.24057582],
               [2.92658295, 5.09667923],
               [3.43493543, 5.34080741],
               [3.20367116, 4.85924759],
               [0.10511804, 4.72916344],
               [1.40597916, 5.06636822],
               [2.24185052, 4.9244617],
               [1.36678395, 5.26161095],
               [1.70725482, 4.04231479],
               [1.91909566, 5.57848447],
               [1.60156731, 4.64453012],
               [0.37963437, 5.26194729],
               [2.02134502, 4.41267445],
               [1.12036737, 5.20880747],
               [2.26901428, 4.61818883],
               [0.24512713, 5.74019237],
               [5.12857843, 5.01149793],
               [1.84419981, 5.03153948],
               [2.32558253, 4.74867962],
               [1.52334113, 4.87916159],
               [1.02285128, 5.0105065],
               [5.85382737, 5.00752482],
               [2.20321658, 4.94516379],
               [1.20099981, 4.57829763],
               [1.02062703, 4.62991119],
               [1.60493227, 5.13663139],
               [0.47647355, 5.13535977],
               [0.3639172, 4.73332823],
               [0.31319845, 5.54694644],
               [2.28664839, 5.0076699],
               [2.15460139, 5.46282959],
               [2.05288518, 4.77958559],
               [4.88804332, 5.50670795],
               [2.40304747, 5.08147326],
               [2.56869453, 5.20687886],
               [1.82975993, 4.59657288],
               [0.54845223, 5.0267298],
               [3.17109619, 5.5946452],
               [3.04202069, 5.00758373],
               [2.40427775, 5.0258707],
               [0.17783466, 5.29765032],
               [1.61428678, 5.22287414],
               [2.30097798, 4.97235844],
               [3.90779317, 5.09464676],
               [2.05670542, 5.23391326],
               [1.38133497, 5.00194962],
               [1.16074178, 4.67727927],
               [1.72818199, 5.36028437],
               [3.20360621, 0.7222149],
               [3.06192918, 1.5719211],
               [4.01714917, 1.16070647],
               [1.40260822, 1.08726536],
               [4.08164951, 0.87200343],
               [3.15273081, 0.98155871],
               [3.45186351, 0.42784083],
               [3.85384314, 0.7920479],
               [1.57449255, 1.34811126],
               [4.72372078, 0.62044136],
               [2.87961084, 0.75413741],
               [0.96791348, 1.16166819],
               [1.53178107, 1.10054852],
               [4.13835915, 1.24780979],
               [3.16109021, 1.29422893],
               [2.95177039, 0.89583143],
               [3.27844295, 1.75043926],
               [2.1270185, 0.95672042],
               [3.32648885, 1.28019066],
               [2.54371489, 0.95732716],
               [3.233947, 1.08202324],
               [4.43152976, 0.54041],
               [3.56478625, 1.11764714],
               [4.25588482, 0.90643957],
               [4.05386581, 0.53291862],
               [3.08970176, 1.08814448],
               [2.84734459, 0.26759253],
               [3.63586049, 1.12160194],
               [1.95538864, 1.32156857],
               [2.88384005, 0.80454506],
               [3.48444387, 1.13551448],
               [3.49798412, 1.10046402],
               [2.45575934, 0.78904654],
               [3.2038001, 1.02728075],
               [3.00677254, 0.62519128],
               [1.96547974, 1.2173076],
               [2.17989333, 1.30879831],
               [2.61207029, 0.99076856],
               [3.95549912, 0.83269299],
               [3.64846482, 1.62849697],
               [4.18450011, 0.45356203],
               [3.7875723, 1.45442904],
               [3.30063655, 1.28107588],
               [3.02836363, 1.35635189],
               [3.18412176, 1.41410799],
               [4.16911897, 0.20581038],
               [2.24024211, 1.14876237],
               [3.91596068, 1.01225774],
               [2.96979716, 1.01210306],
               [1.12993856, 0.77085284],
               [2.71730799, 0.48697555],
               [3.1189017,  0.69438336],
               [2.4051802,  1.11778123],
               [2.95818429, 1.01887096],
               [1.65456309, 1.18631175],
               [2.39775807, 1.24721387],
               [2.28409305, 0.64865469],
               [2.79588724, 0.99526664],
               [3.41156277, 1.1596363],
               [3.50663521, 0.73878104],
               [3.93616029, 1.46202934],
               [3.90206657, 1.27778751],
               [2.61036396, 0.88027602],
               [4.37271861, 1.02914092],
               [3.08349136, 1.19632644],
               [2.1159935, 0.7930365],
               [2.15653404, 0.40358861],
               [2.14491101, 1.13582399],
               [1.84935524, 1.02232644],
               [4.1590816, 0.61720733],
               [2.76494499, 1.43148951],
               [3.90561153, 1.16575315],
               [2.54071672, 0.98392516],
               [4.27783068, 1.1801368],
               [3.31058167, 1.03124461],
               [2.15520661, 0.80696562],
               [3.71363659, 0.45813208],
               [3.54010186, 0.86446135],
               [1.60519991, 1.1098053],
               [1.75164337, 0.68853536],
               [3.12405123, 0.67821757],
               [2.37198785, 1.42789607],
               [2.53446019, 1.21562081],
               [3.6834465, 1.22834538],
               [3.2670134, 0.32056676],
               [3.94159139, 0.82577438],
               [3.2645514, 1.3836869],
               [4.30471138, 1.10725995],
               [2.68499376, 0.35344943],
               [3.12635184, 1.2806893],
               [2.94294356, 1.02825076],
               [3.11876541, 1.33285459],
               [2.02358978, 0.44771614],
               [3.62202931, 1.28643763],
               [2.42865879, 0.86499285],
               [2.09517296, 1.14010491],
               [5.29239452, 0.36873298],
               [2.07291709, 1.16763851],
               [0.94623208, 0.24522253],
               [2.73911908, 1.10072284],
               [6.00506534, 2.72784171],
               [6.05696411, 2.94970433],
               [6.77012767, 3.21411422],
               [5.64034678, 2.69385282],
               [5.63325403, 2.99002339],
               [6.17443157, 3.29026488],
               [7.24694794, 2.96877424],
               [5.58162906, 3.33510375],
               [5.3627205, 3.14681192],
               [4.70775773, 2.78710869],
               [7.42892098, 3.4667949],
               [6.64107248, 3.05998738],
               [6.37473652, 2.56253059],
               [7.28780324, 2.75179885],
               [6.20295231, 2.67856179],
               [5.38736041, 2.26737346],
               [5.6673103, 2.96477867],
               [6.59702155, 3.07082376],
               [7.75660559, 3.15604465],
               [6.63262745, 3.14799183],
               [5.76634959, 3.14271707],
               [5.99423154, 2.75707858],
               [6.37870407, 2.65022321],
               [5.74036233, 3.10391306],
               [4.61652442, 2.79320715],
               [5.33533999, 3.03928694],
               [5.37293912, 2.81684776],
               [5.03611162, 2.92486087],
               [5.52908677, 3.33681576],
               [6.05086942, 2.80702594],
               [5.132009, 2.19812195],
               [5.73284945, 2.87738132],
               [6.78110732, 3.05676866],
               [6.44834449, 3.35299225],
               [6.39941482, 2.89756948],
               [5.86067925, 2.99577129],
               [5.44765183, 3.16560945],
               [5.36708111, 3.19502552],
               [5.88735565, 3.34615566],
               [3.96162465, 2.72025046],
               [6.28438193, 3.17360643],
               [4.20584789, 2.81647368],
               [5.32615581, 3.03314047],
               [7.17135204, 3.4122727],
               [7.4949275, 2.84018754],
               [7.39807241, 3.48487031],
               [5.02432984, 2.98683179],
               [5.31712478, 2.81741356],
               [5.87655237, 3.21661109],
               [6.03762833, 2.68303512],
               [5.91280273, 2.85631938],
               [6.69451358, 2.89056083],
               [6.01017978, 2.72401338],
               [6.92721968, 3.19960026],
               [6.33559522, 3.30864291],
               [6.24257071, 2.79179269],
               [5.57812294, 3.24766016],
               [6.40773863, 2.67554951],
               [6.80029526, 3.17579578],
               [7.21684033, 2.72896575],
               [6.5110074, 2.72731907],
               [4.60630534, 3.329458],
               [7.65503226, 2.87095628],
               [5.50295759, 2.62924634],
               [6.63060699, 3.01502301],
               [4.45928006, 2.68478445],
               [8.20339815, 2.41693495],
               [4.95679428, 2.89776297],
               [5.37052667, 2.44954813],
               [5.69797866, 2.94977132],
               [6.27376271, 2.24256036],
               [5.05274526, 2.75692163],
               [6.88575584, 2.88845269],
               [4.1877442, 2.89283463],
               [5.97510328, 3.0259191],
               [6.09457129, 2.61867975],
               [5.72395697, 3.04454219],
               [4.37249767, 3.05488217],
               [6.29206262, 2.77573856],
               [5.14533035, 4.13225692],
               [6.5870565, 3.37508345],
               [5.78769095, 3.29255127],
               [6.72798098, 3.0043983],
               [6.64078939, 2.41068839],
               [6.23228878, 2.72850902],
               [6.21772724, 2.80994633],
               [5.78116301, 3.07987787],
               [6.62447253, 2.74453743],
               [5.19590823, 3.06972937],
               [5.87177181, 3.2551773],
               [5.89562099, 2.89843977],
               [5.6175432, 2.5975071],
               [5.63176103, 3.04758747],
               [5.50258659, 3.11869075],
               [6.48212628, 2.5508514],
               [7.30278708, 3.38015979],
               [6.99198434, 2.98706729],
               [4.8255341, 2.77961664],
               [6.11768055, 2.85475655],
               [0.94048944, 5.71556802]])

