// ============================================================================
//  TestingMacro.C  -  lepton pair selection (J/psi) + e/mu classification (ONNX RF)
//
//  Run (ACLiC compilation recommended):
//      root -l -b -q 'TestingMacro.C+'
//
//  Output:
//      Plots/TestingControl/Histograms<name>.root    - all histograms (directories)
//          kinematics/   - lepton pair properties
//          classifier/   - classifier response
//          clusters_all/ - topoclusters for all tracks (before classification)
//          efficiency/   - topocluster-finding efficiency vs pT, eta, (pT,eta)
//                          (num/den histograms + ratios from TH1::Divide, "B")
//          matching/     - control plots of track-cluster matching (dR, dEta vs dPhi)
//          cutflow/      - number of events passing each cut (cumulative)
//          rapidity/     - cluster features and RF response vs pair rapidity
//                          (TH2D + TProfile; prefixes all_, electron_, muon_)
//          electron/     - histograms of tracks classified as electrons
//          muon/         - histograms of tracks classified as muons
// ============================================================================

#include <onnxruntime_cxx_api.h>

#include <TChain.h>
#include <TDirectory.h>
#include <TError.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <TMath.h>
#include <TROOT.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TVector2.h>

#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <string>

// ----------------------------------------------------------------------------
//  Configuration
// ----------------------------------------------------------------------------
namespace cfg {
constexpr double kMuonMass    = 0.1056583755;  // GeV
constexpr double kMaxAbsEta   = 2.5;
constexpr double kMinTrackPt  = 1.0;           // GeV
constexpr double kMaxD0       = 1.0;           // same unit as the track_d0 branch (usually mm)
constexpr double kMaxPairPt   = 0.2;           // GeV
constexpr double kMinPairMass = 2.0;           // GeV
constexpr double kMaxPairMass = 3.5;           // GeV
constexpr double kConeDR      = 0.8;           // topocluster matching cone
constexpr float  kProbCut     = 0.5f;          // classifier threshold
constexpr int    kNYBins      = 5;             // pair |y| bins, 0.5 wide
constexpr int    kPrintEvery  = 200000;

// topocluster selection
constexpr double kMaxClusterEta = 4.9;
constexpr double kMinClusterPt  = 0.1;         // GeV

// Efficiency binning
constexpr int    kEffPtBins   = 40;            // 1D vs pT
constexpr double kEffPtMin    = 1.0;           // GeV
constexpr double kEffPtMax    = 1.8;           // GeV
constexpr int    kEffEtaBins  = 50;            // 1D vs eta
constexpr int    kEff2DPtBins = 50;            // 2D (pT, eta) - coarser, fewer entries per bin

// Rapidity binning
constexpr int    kRapBins     = 24;            // signed y of the pair
constexpr double kRapMax      = 3.0;
}  // namespace cfg

enum Species { kElectron = 0, kMuon = 1, kUnknown = 2 };

// Cutflow stages (cumulative)
enum Cut {
   kCutAll = 0,      // all events
   kCutTrigger,      // HLT trigger
   kCutTwoTracks,    // exactly 2 tracks after |eta|, pT, d0, Loose
   kCutOppCharge,    // opposite charge
   kCutPairPt,       // pair pT < kMaxPairPt
   kCutPairMass,     // kMinPairMass <= M < kMaxPairMass
   kCutClassified,   // classified as e or mu (not ambiguous / no cluster)
   kNCuts
};

const char* const kCutLabels[kNCuts] = {
   "All events", "Trigger", "2 tracks (eta,pT,d0,Loose)", "Opposite charge",
   "Pair pT", "Pair mass", "Classified e/#mu"
};

// ----------------------------------------------------------------------------
//  Helper structures
// ----------------------------------------------------------------------------

// Result of matching a topocluster to a track
struct ClusterVars {
   bool  found    = false;
   int   nMatched = 0;
   float pt       = 0;
   float f        = -1;
   float emProb   = -1;
   float lambda   = -1;
   float lambda2  = -1;
   float r2       = -1;
};

TH1D* Book(TDirectory* dir, const std::string& name, const std::string& title, int n, double lo, double hi)
{
   dir->cd();
   return new TH1D(name.c_str(), title.c_str(), n, lo, hi);
}

// Set of topocluster histograms
struct ClusterHists {
   TH1D *count, *pt, *f, *em, *lambda, *lambda2, *radius;

   void Book(TDirectory* d, const std::string& prefix = "")
   {
      count   = ::Book(d, prefix + "count",   "Matched clusters", 15, -0.5, 14.5);
      pt      = ::Book(d, prefix + "pt",      "Cluster p_{T}",    100, 0, 3.5);
      f       = ::Book(d, prefix + "f",       "F = p_{T}^{cl}/p_{T}^{trk}", 100, 0, 4);
      em      = ::Book(d, prefix + "emProb",  "EM probability",   40, -0.1, 1.1);
      lambda  = ::Book(d, prefix + "lambda",  "Lambda",           40, 0, 4);
      lambda2 = ::Book(d, prefix + "lambda2", "Lambda2",          40, 0, 600);
      radius  = ::Book(d, prefix + "radius",  "Radius",           40, 0, 100);
   }

   void Fill(const ClusterVars& c)
   {
      count->Fill(c.nMatched);
      if (!c.found) return;
      pt->Fill(c.pt);
      f->Fill(c.f);
      em->Fill(c.emProb);
      lambda->Fill(c.lambda / 1000);
      lambda2->Fill(c.lambda2 / 1000);
      radius->Fill(c.r2 / 1000);
   }
};

// Topocluster-finding efficiency for a single track.
struct EffHists {
   TH1D *numPt, *denPt, *numEta, *denEta;
   TH2D *numPtEta, *denPtEta;

   void Book(TDirectory* d)
   {
      d->cd();
      const double etaMax = 3.0;

      numPt  = new TH1D("num_pt",  "Tracks with topocluster;p_{T} [GeV];Tracks",
                        cfg::kEffPtBins, cfg::kEffPtMin, cfg::kEffPtMax);
      denPt  = new TH1D("den_pt",  "All tracks;p_{T} [GeV];Tracks",
                        cfg::kEffPtBins, cfg::kEffPtMin, cfg::kEffPtMax);
      numEta = new TH1D("num_eta", "Tracks with topocluster;#eta;Tracks",
                        cfg::kEffEtaBins, -etaMax, etaMax);
      denEta = new TH1D("den_eta", "All tracks;#eta;Tracks",
                        cfg::kEffEtaBins, -etaMax, etaMax);
      numPtEta = new TH2D("num_pt_eta", "Tracks with topocluster;p_{T} [GeV];#eta",
                          cfg::kEff2DPtBins, cfg::kEffPtMin, cfg::kEffPtMax,
                          cfg::kEffEtaBins, -etaMax, etaMax);
      denPtEta = new TH2D("den_pt_eta", "All tracks;p_{T} [GeV];#eta",
                          cfg::kEff2DPtBins, cfg::kEffPtMin, cfg::kEffPtMax,
                          cfg::kEffEtaBins, -etaMax, etaMax);

      for (TH1* h : {static_cast<TH1*>(numPt), static_cast<TH1*>(denPt),
                     static_cast<TH1*>(numEta), static_cast<TH1*>(denEta),
                     static_cast<TH1*>(numPtEta), static_cast<TH1*>(denPtEta)})
         h->Sumw2();
   }

   void Fill(const TLorentzVector& p, bool found)
   {
      denPt->Fill(p.Pt());
      denEta->Fill(p.Eta());
      denPtEta->Fill(p.Pt(), p.Eta());
      if (!found) return;
      numPt->Fill(p.Pt());
      numEta->Fill(p.Eta());
      numPtEta->Fill(p.Pt(), p.Eta());
   }

   void Finalize(TDirectory* d)
   {
      Ratio(d, numPt,    denPt,    "eff_pt",     "Topocluster efficiency;p_{T} [GeV];Efficiency");
      Ratio(d, numEta,   denEta,   "eff_eta",    "Topocluster efficiency;#eta;Efficiency");
      Ratio(d, numPtEta, denPtEta, "eff_pt_eta", "Topocluster efficiency;p_{T} [GeV];#eta");
   }

   // Integrated efficiency over all tracks
   double Total() const
   {
      const double den = denEta->GetEntries();
      return den > 0 ? numEta->GetEntries() / den : 0.;
   }

private:
   template <class H>
   static void Ratio(TDirectory* d, const H* num, const H* den, const char* name, const char* title)
   {
      auto* eff = static_cast<H*>(num->Clone(name));
      eff->SetDirectory(d);
      eff->SetTitle(title);
      eff->Divide(num, den, 1., 1., "B");  // binomial errors
   }
};

// Cluster features and RF response as a function of pair rapidity
struct RapidityHists {
   struct Var { TH2D* h2; TProfile* prof; };
   Var pt, f, em, lambda, lambda2, radius, resp, respMean;

   void Book(TDirectory* d, const std::string& p)
   {
      pt       = BookVar(d, p, "clusterPt",    "p_{T}^{cl} [GeV]",           50, 0, 3.5);
      f        = BookVar(d, p, "f",            "F = p_{T}^{cl}/p_{T}^{trk}", 100, 0, 4);
      em       = BookVar(d, p, "emProb",       "EM probability",             40, -0.1, 1.1);
      lambda   = BookVar(d, p, "lambda",       "Lambda",                     40, 0, 4);
      lambda2  = BookVar(d, p, "lambda2",      "Lambda2",                    40, 0, 600);
      radius   = BookVar(d, p, "radius",       "Radius",                     40, 0, 100);
      resp     = BookVar(d, p, "response",     "RF response (single track)", 50, 0, 1);
      respMean = BookVar(d, p, "responseMean", "RF response (pair mean)",    50, 0, 1);
   }

   // Per track: cluster features + single-track response
   void Fill(const ClusterVars& c, double y, std::optional<float> r)
   {
      if (!c.found) return;
      FillVar(pt,      y, c.pt);
      FillVar(f,       y, c.f);
      FillVar(em,      y, c.emProb);
      FillVar(lambda,  y, c.lambda  / 1000);   // same scaling as ClusterHists
      FillVar(lambda2, y, c.lambda2 / 1000);
      FillVar(radius,  y, c.r2      / 1000);
      if (r) FillVar(resp, y, *r);
   }

   // Per pair: mean response
   void FillPair(double y, float meanResp)
   {
      if (meanResp >= 0) FillVar(respMean, y, meanResp);
   }

private:
   static Var BookVar(TDirectory* d, const std::string& p, const std::string& name,
                      const char* ytitle, int nb, double lo, double hi)
   {
      d->cd();
      Var v;
      v.h2   = new TH2D((p + name + "_vs_y").c_str(), Form(";y_{pair};%s", ytitle),
                        cfg::kRapBins, -cfg::kRapMax, cfg::kRapMax, nb, lo, hi);
      v.prof = new TProfile((p + name + "_prof_y").c_str(), Form(";y_{pair};<%s>", ytitle),
                            cfg::kRapBins, -cfg::kRapMax, cfg::kRapMax);
      return v;
   }
   static void FillVar(Var& v, double y, double val) { v.h2->Fill(y, val); v.prof->Fill(y, val); }
};

constexpr int kAllIdx = 2;   // index of the "all tracks" set (electron/muon sets use kElectron/kMuon)

// Set of histograms for one lepton species
struct LeptonHists {
   TH1D *eta, *energy, *pt, *phi;
   TH1D *pixelHits, *trtHits, *sctHits, *pixeldEdX, *d0;
   TH1D *dimass;
   std::array<TH1D*, cfg::kNYBins> dimassY;
   ClusterHists clusters;

   void Book(TDirectory* d, const std::string& label)
   {
      eta       = ::Book(d, "eta",       label + " #eta",      50, -3, 3);
      energy    = ::Book(d, "energy",    label + " energy",    50, 0, 12);
      pt        = ::Book(d, "pt",        label + " p_{T}",     50, 0, 2.8);
      phi       = ::Book(d, "phi",       label + " #phi",      30, -TMath::Pi(), TMath::Pi());
      d0        = ::Book(d, "d0",        "Track d_{0};d_{0};Tracks", 100, -cfg::kMaxD0, cfg::kMaxD0);
      pixelHits = ::Book(d, "pixelHits", "PixelHits",          11, -0.5, 10.5);
      trtHits   = ::Book(d, "trtHits",   "TRTHits",            51, -1, 50);
      sctHits   = ::Book(d, "sctHits",   "SCTHits",            12, 2.5, 14.5);
      pixeldEdX = ::Book(d, "pixeldEdX", "PixeldEdX",          100, 0, 2);
      dimass    = ::Book(d, "dimass",    "Lepton pair mass",   50, 2.0, 3.5);
      for (int b = 0; b < cfg::kNYBins; ++b)
         dimassY[b] = ::Book(d, Form("dimass_y%d", b),
                             Form("Lepton pair mass: |y| %g - %g", b * 0.5, b * 0.5 + 0.5),
                             40, 2.5, 3.5);
      clusters.Book(d, "clu_");
   }

   void FillTrack(const TLorentzVector& p, float pix, float trt, float sct, float dedx, float d0val)
   {
      energy->Fill(p.E());
      eta->Fill(p.Eta());
      pt->Fill(p.Pt());
      phi->Fill(p.Phi());
      d0->Fill(d0val);
      pixelHits->Fill(pix);
      trtHits->Fill(trt);
      sctHits->Fill(sct);
      pixeldEdX->Fill(dedx);
   }
};

// ONNX session wrapper (e/mu Random Forest)
class RFClassifier {
public:
   explicit RFClassifier(const char* path) : fEnv(ORT_LOGGING_LEVEL_WARNING, "rf"), fSession(nullptr)
   {
      Ort::SessionOptions opts;
      opts.SetIntraOpNumThreads(1);
      fSession = Ort::Session(fEnv, path, opts);

      Ort::AllocatorWithDefaultOptions alloc;
      fInputName = fSession.GetInputNameAllocated(0, alloc).get();
   }

   // Returns P(muon)
   float Predict(std::array<float, 5>& x)
   {
      auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      std::array<int64_t, 2> shape{1, 5};
      Ort::Value input = Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), shape.data(), shape.size());

      const char* inNames[]  = {fInputName.c_str()};
      const char* outNames[] = {"label", "probabilities"};
      auto out = fSession.Run(Ort::RunOptions{nullptr}, inNames, &input, 1, outNames, 2);
      return out[1].GetTensorMutableData<float>()[1];
   }

private:
   Ort::Env     fEnv;
   Ort::Session fSession;
   std::string  fInputName;
};

// Pair |y| bin; -1 if out of range
inline int YBin(double y)
{
   const int b = int(std::fabs(y) / 0.5);
   return b < cfg::kNYBins ? b : -1;
}

// ----------------------------------------------------------------------------
//  Main function
// ----------------------------------------------------------------------------
void TestingMacro()
{
    //const std::string name = "TrueData"; const std::string inputFile = "../Data/data23_2trk_moreTCvars.root";
    const std::string name = "MC_ee";    const std::string inputFile = "../Data/mc_jpsi_ee.root";
    //const std::string name = "MC_mumu";    const std::string inputFile = "../Data/mc_jpsi_mumu.root";
    gROOT->SetBatch(kTRUE);
    gErrorIgnoreLevel = kError;

   // ---------------- Output file ----------------
    TFile histFile(Form("Plots/TestingControl/Histograms%s.root", name.c_str()), "RECREATE");

   // ---------------- Histograms ----------------
    TDirectory* dKin   = histFile.mkdir("kinematics");
    TDirectory* dClf   = histFile.mkdir("classifier");
    TDirectory* dAll   = histFile.mkdir("clusters_all");
    TDirectory* dEff   = histFile.mkdir("efficiency");
    TDirectory* dEl    = histFile.mkdir("electron");
    TDirectory* dMu    = histFile.mkdir("muon");
    TDirectory* dMatch = histFile.mkdir("matching");
    TDirectory* dCut   = histFile.mkdir("cutflow");
    TDirectory* dRap   = histFile.mkdir("rapidity");

    TH1D* hCutflow = Book(dCut, "cutflow", "Cutflow;;Events", kNCuts, 0, kNCuts);
    for (int b = 0; b < kNCuts; ++b) hCutflow->GetXaxis()->SetBinLabel(b + 1, kCutLabels[b]);

    TH1D* hPairPt    = Book(dKin, "pt",       "Transverse momentum of lepton pair", 50, 0, 0.5);
    TH1D* hPairY     = Book(dKin, "rapidity", "Rapidity of lepton pair",            50, -3, 3);
    TH1D* hPairMass  = Book(dKin, "dimass",   "Lepton pair mass",                   50, 2.5, 3.5);
    TH1D* hDeltaPhi  = Book(dKin, "deltaPhi", "#Delta#phi [deg]",                   40, 160, 180);
    TH1D* hTrackD0   = Book(dKin, "track_d0", "Selected track d_{0};d_{0};Tracks",  100, -cfg::kMaxD0, cfg::kMaxD0);
    TH1D* hYMid = Book(dKin, "rapidity_mid", "Pair rapidity, 0.30<score<0.70", 50, -3, 3);

    TH1D* hResponse     = Book(dClf, "response",          "RF response to experimental data", 50, 0, 1);
    TH1D* hResponseMean = Book(dClf, "response_mean",     "RF mean response to experimental data", 50, 0, 1);
    TH1D* hRespOneFound = Book(dClf, "responseOneFound",  "Response (one track matched)",     100, 0, 1);
    TH1D* hRespAmbig    = Book(dClf, "responseAmbiguous", "Response (ambiguous pair)",        100, 0, 1);

    ClusterHists allClusters;
    allClusters.Book(dAll);

    EffHists eff;
    eff.Book(dEff);

    // Control histograms of the track-cluster matching (filled inside matchCluster)
    TH1D* hDR      = Book(dMatch, "dR",      "#DeltaR(track, cluster), all clusters passing the cut;#DeltaR;Clusters", 120, 0, 6);
    TH1D* hPt      = Book(dMatch, "dPt",     "Transverse momentum cluster;p_{T};Clusters", 120, 0, 6);
    TH1D* hDRBest  = Book(dMatch, "dR_best", "#DeltaR(track, selected cluster);#DeltaR;Tracks", 100, 0, cfg::kConeDR);
    dMatch->cd();
    TH2D* hDEtaDPhi = new TH2D("dEta_dPhi", "Track - cluster;#Delta#eta;#Delta#phi;Clusters",
                               120, -cfg::kConeDR, cfg::kConeDR, 64, -cfg::kConeDR, cfg::kConeDR);

    std::array<LeptonHists, 2> lepHists;
    lepHists[kElectron].Book(dEl, "Electron");
    lepHists[kMuon].Book(dMu, "Muon");

    // Cluster features and response vs pair rapidity: all tracks / electrons / muons
    std::array<RapidityHists, 3> rap;
    rap[kAllIdx  ].Book(dRap, "all_");
    rap[kElectron].Book(dRap, "electron_");
    rap[kMuon    ].Book(dRap, "muon_");

   // ---------------- Classifier ----------------
    //RFClassifier classifier("../ONNX/RFMuonElectron.onnx");
    RFClassifier classifier("../ONNX/XGBMuonElectron.onnx");

   // ---------------- Input ----------------
    TChain chain("G2TauTree");
    chain.Add(inputFile.c_str());
    TTreeReader reader(&chain);

    TTreeReaderArray<bool> TriggerPass(reader, "passed_HLT_mb_excl_1trk5_pt1_hi_FgapAC5_L1TRT_VTE20");

    TTreeReaderArray<float> TrackEta   (reader, "track_eta");
    TTreeReaderArray<float> TrackPhi   (reader, "track_phi");
    TTreeReaderArray<float> TrackPt    (reader, "track_pt");
    TTreeReaderArray<float> TrackD0    (reader, "track_d0");
    TTreeReaderArray<int>   TrackCharge(reader, "track_charge");
    TTreeReaderArray<bool>  TrackLoose (reader, "track_is_LoosePrimary");

    TTreeReaderArray<float> TrackPixeldEdX(reader, "track_PixeldEdX");
    TTreeReaderArray<float> TrackPixelHits(reader, "track_PixelHits");
    TTreeReaderArray<float> TrackSCTHits  (reader, "track_SCTHits");
    TTreeReaderArray<float> TrackTRTHits  (reader, "track_TRTHits");

    TTreeReaderArray<float> TopoEta    (reader, "topo_cluster_eta");
    TTreeReaderArray<float> TopoPhi    (reader, "topo_cluster_phi");
    TTreeReaderArray<float> TopoPt     (reader, "topo_cluster_pt");
    TTreeReaderArray<float> TopoLambda (reader, "topo_cluster_lambda");
    TTreeReaderArray<float> TopoLambda2(reader, "topo_cluster_lambda2");
    TTreeReaderArray<float> TopoR2     (reader, "topo_cluster_r2");
    TTreeReaderArray<float> TopoEMProb (reader, "topo_cluster_EM_prob");
    TTreeReaderArray<bool>  TopoPass   (reader, "topo_cluster_pass_sig_cut");

   // Topocluster matching: the highest-pT cluster in a cone of dR < kConeDR.
    auto matchCluster = [&](const TLorentzVector& p, bool fillControl = false) {
        ClusterVars c;
        int best = -1;
        double bestDR = -1;
        for (int t = 0; t < int(TopoPt.GetSize()); ++t) {
            if (std::fabs(TopoEta[t]) >= cfg::kMaxClusterEta || TopoPt[t] <= cfg::kMinClusterPt) continue;
            if (!TopoPass[t]) continue;
            const double dPhi = TVector2::Phi_mpi_pi(p.Phi() - TopoPhi[t]);
            const double dEta = p.Eta() - TopoEta[t];
            const double dR = std::hypot(dEta, dPhi);

            if (fillControl) {
                hDR->Fill(dR);
                hPt->Fill(TopoPt[t]);
                hDEtaDPhi->Fill(dEta, dPhi);
            }
            if (dR >= cfg::kConeDR) continue;

            ++c.nMatched;

            if (TopoPt[t] > c.pt) { c.pt = TopoPt[t]; best = t; bestDR = dR; }
        }
        if (best >= 0) {
            if (fillControl) hDRBest->Fill(bestDR);
            c.found   = true;
            c.f       = TopoPt[best] / p.Perp();
            c.emProb  = TopoEMProb[best];
            c.lambda  = TopoLambda[best];
            c.lambda2 = TopoLambda2[best];
            c.r2      = TopoR2[best];
        }
        return c;
    };

   // ---------------- Counters ----------------
    long  eventID = 0;
    std::array<long, 2> leptonCount{0, 0};
    float allParticles = 0, zeroFound = 0, oneFound = 0, twoFound = 0, ambiguous = 0;

   // ---------------- Event loop ----------------
    while (reader.Next()) {
        if (eventID % cfg::kPrintEvery == 0) std::cout << "Processing " << eventID << " event..." << std::endl;
        ++eventID;
        hCutflow->Fill(kCutAll);

        // --- Trigger ---
        if (!TriggerPass[0]) continue;
        hCutflow->Fill(kCutTrigger);

        // --- Track selection: |eta|<2.5, pT>1 GeV, |d0|<1, Loose; exactly 2 selected tracks ---
        int idx[2] = {-1, -1};
        int nSel = 0;
        for (int i = 0; i < int(TrackPt.GetSize()); ++i) {
            if (std::fabs(TrackEta[i]) >= cfg::kMaxAbsEta || TrackPt[i] <= cfg::kMinTrackPt) continue;
            if (std::fabs(TrackD0[i]) >= cfg::kMaxD0 || !TrackLoose[i]) continue;
            if (nSel < 2) idx[nSel] = i;
            ++nSel;
        }
        if (nSel != 2) continue;
        hCutflow->Fill(kCutTwoTracks);

        if (TrackCharge[idx[0]] == TrackCharge[idx[1]]) continue;
        hCutflow->Fill(kCutOppCharge);

        std::array<TLorentzVector, 2> trk;
        for (int k = 0; k < 2; ++k)
            trk[k].SetPtEtaPhiM(TrackPt[idx[k]], TrackEta[idx[k]], TrackPhi[idx[k]], cfg::kMuonMass);

        // --- Pair selection: pT < 0.2 GeV, 2.0 <= M < 3.5 GeV ---
        const TLorentzVector pair = trk[0] + trk[1];
        hPairPt->Fill(pair.Perp());

        if (pair.Perp() >= cfg::kMaxPairPt) continue;
        hCutflow->Fill(kCutPairPt);
        if (pair.M() >= cfg::kMaxPairMass || pair.M() < cfg::kMinPairMass) continue;
        hCutflow->Fill(kCutPairMass);

        const double yPair = pair.Rapidity();

        hPairPt->Fill(pair.Perp());
        hPairY->Fill(yPair);
        hPairMass->Fill(pair.M());
        hDeltaPhi->Fill(std::fabs(TVector2::Phi_mpi_pi(trk[0].Phi() - trk[1].Phi())) * TMath::RadToDeg());
        for (int k = 0; k < 2; ++k) hTrackD0->Fill(TrackD0[idx[k]]);

        // --- e/mu classification for each track ---
        std::array<std::optional<float>, 2> response;
        for (int i = 0; i < 2; ++i) {
            ++allParticles;
            const ClusterVars cl = matchCluster(trk[i], true);   // fill control histograms here only
            allClusters.Fill(cl);
            eff.Fill(trk[i], cl.found);   // topocluster-finding efficiency (single track)
            if (!cl.found) continue;

            std::array<float, 5> x = {cl.f, cl.emProb, cl.lambda, cl.lambda2, cl.r2};
            response[i] = classifier.Predict(x);
            hResponse->Fill(*response[i]);
            rap[kAllIdx].Fill(cl, yPair, response[i]);
        }

        // --- Decision: mean response from available tracks ---
        const int nFound = int(response[0].has_value()) + int(response[1].has_value());
        Species species  = kUnknown;
        float meanResp   = -1.f;

        if (nFound == 0) {
            zeroFound += 2;
        } else {
            const float mean = (response[0].value_or(0.f) + response[1].value_or(0.f)) / nFound;
            if (mean > 0.30f && mean < 0.70f) hYMid->Fill(yPair);
            meanResp = mean;
            rap[kAllIdx].FillPair(yPair, mean);
            if (nFound == 1) { oneFound += 2; hRespOneFound->Fill(mean); }
            else             { twoFound += 2; }

            if      (mean < cfg::kProbCut) species = kElectron;
            else if (mean > cfg::kProbCut) species = kMuon;
            else if (nFound == 2) { ambiguous += 2; hRespAmbig->Fill(*response[0]); }
        }
        if (species == kUnknown) continue;
        hCutflow->Fill(kCutClassified);

        // --- Fill histograms for the classified pair ---
        LeptonHists& H = lepHists[species];
        H.dimass->Fill(pair.M());
        if (const int yb = YBin(yPair); yb >= 0) H.dimassY[yb]->Fill(pair.M());
        rap[species].FillPair(yPair, meanResp);
        hResponseMean->Fill(meanResp);

        for (int i = 0; i < 2; ++i) {
            const int t = idx[i];
            H.FillTrack(trk[i], TrackPixelHits[t], TrackTRTHits[t], TrackSCTHits[t],
                        TrackPixeldEdX[t], TrackD0[t]);

            const ClusterVars cl = matchCluster(trk[i]);
            H.clusters.Fill(cl);
            rap[species].Fill(cl, yPair, response[i]);
            if (cl.found) ++leptonCount[species];
        }
    }

   // ---------------- Efficiencies (binomial errors) ----------------
    eff.Finalize(dEff);

   // ---------------- Summary ----------------
    const float denom = allParticles > 0 ? allParticles : 1.f;
    std::cout << "===========================\n"
              << "End of file: " << name << "\n"
              << "Number of events:    " << eventID << "\n"
              << "Number of electrons: " << leptonCount[kElectron] << "\n"
              << "Number of muons:     " << leptonCount[kMuon] << "\n"
              << "===========================\n"
              << "Found count:\n"
              << "   Two particles:   " << twoFound  << "  ratio " << twoFound  / denom << "\n"
              << "   One particle:    " << oneFound  << "  ratio " << oneFound  / denom << "\n"
              << "   Zero particles:  " << zeroFound << "  ratio " << zeroFound / denom << "\n"
              << "   Ambiguous pairs: " << ambiguous << "\n"
              << "   Topocluster efficiency (single track, integrated): " << eff.Total() << "\n"
              << "===========================\n"
              << "Cutflow:\n";
    for (int b = 1; b <= kNCuts; ++b) {
        const double n    = hCutflow->GetBinContent(b);
        const double prev = b > 1 ? hCutflow->GetBinContent(b - 1) : n;
        std::cout << "   " << kCutLabels[b - 1] << ": " << n
                  << "  (" << (prev > 0 ? 100. * n / prev : 0.) << "% of previous)\n";
    }
    std::cout << "===========================" << std::endl;

   // ---------------- Save output ----------------
    histFile.Write();
    histFile.Close();
}