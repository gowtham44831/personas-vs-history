import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})

# Figure 1: matched vs mismatched personas (test window)
ph_rows=["Apple premium","Budget hunters","Frequent multi-device","Samsung loyalists"]
ph=np.array([[0.697,0.547,0.625,0.455],[0.566,0.531,0.583,0.522],[0.518,0.504,0.555,0.564],[0.501,0.560,0.600,0.669]])
vg_rows=["Budget Nintendo","Frequent multi-platform","Nintendo loyalists","PC gaming gear","PlayStation","Xbox"]
vg=np.array([[0.521,0.560,0.547,0.549,0.510,0.496],[0.503,0.596,0.536,0.557,0.515,0.519],[0.514,0.581,0.634,0.525,0.519,0.490],
             [0.468,0.605,0.553,0.614,0.564,0.588],[0.535,0.607,0.467,0.579,0.628,0.514],[0.532,0.596,0.513,0.602,0.566,0.648]])
fig,axes=plt.subplots(2,1,figsize=(6.4,10),gridspec_kw={"height_ratios":[4,6],"hspace":0.55})
for ax,M,rows,title in [(axes[0],ph,ph_rows,"(a) Cell phones & accessories"),(axes[1],vg,vg_rows,"(b) Video games")]:
    im=ax.imshow(M,cmap="Blues",vmin=0.44,vmax=0.70)
    ax.set_xticks(range(len(rows)));ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(rows,rotation=35,ha="right");ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(rows)):
            ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=8,
                    color="white" if M[i,j]>0.60 else "#1f1f1f",fontweight="bold" if i==j else "normal")
        ax.add_patch(plt.Rectangle((i-0.5,i-0.5),1,1,fill=False,ec="#1f1f1f",lw=1.5))
    ax.set_xlabel("Persona used as judge");ax.set_ylabel("Whose purchase");ax.set_title(title,fontsize=10)
    for s in ax.spines.values(): s.set_visible(False)
fig.colorbar(im,ax=axes,shrink=0.8,label="AUC of the bought item")
fig.savefig("fig1_matched.png",dpi=200,bbox_inches="tight")

# Figure 2: history minus persona vs homogeneity (holdout)
ph=[("Apple premium",0.84,0.014,-0.017,0.047),("Samsung loyalists",0.73,-0.006,-0.046,0.033),
    ("Frequent multi-device",0.50,0.121,0.084,0.161),("Budget hunters",0.45,0.090,0.041,0.137)]
vg=[("Nintendo loyalists",0.76,0.043,0.012,0.073),("PC gaming gear",0.72,0.018,-0.022,0.057),("Xbox",0.71,0.047,0.000,0.094),
    ("PlayStation",0.70,0.049,0.007,0.092),("Budget Nintendo",0.53,0.058,0.015,0.102),("Frequent multi-platform",0.42,0.047,0.007,0.087)]
fig,ax=plt.subplots(figsize=(6.4,4.2))
for data,c,m,lab in [(ph,"#2a6fb0","o","Cell phones"),(vg,"#c4532a","s","Video games")]:
    x=np.array([d[1] for d in data]);y=np.array([d[2] for d in data])
    lo=y-np.array([d[3] for d in data]);hi=np.array([d[4] for d in data])-y
    ax.errorbar(x,y,yerr=[lo,hi],fmt=m,color=c,ms=7,lw=1.5,capsize=3,label=lab)
offs={"Apple premium":(6,4),"Samsung loyalists":(6,-10),"Frequent multi-device":(6,2),"Budget hunters":(6,2),
      "Nintendo loyalists":(6,-2),"PC gaming gear":(6,-10),"Xbox":(-26,8),"PlayStation":(-62,-14),"Budget Nintendo":(6,0),"Frequent multi-platform":(6,-8)}
for d in ph+vg: ax.annotate(d[0],(d[1],d[2]),xytext=offs[d[0]],textcoords="offset points",fontsize=7.5,color="#444")
ax.axhline(0,color="#888",lw=1)
ax.set_xlabel("Segment homogeneity (share of purchases for its main device/platform)")
ax.set_ylabel("AUC gain: own history over segment persona")
ax.legend(frameon=False,loc="upper right");ax.grid(axis="y",color="#e5e5e5",lw=0.8)
fig.savefig("fig2_homogeneity.png",dpi=200,bbox_inches="tight")
print("done")
