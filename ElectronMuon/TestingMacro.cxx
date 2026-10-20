// ============================================================================
//  Analysis.C  -  lepton pair selection (J/psi) + e/mu classification (ONNX RF)
//
//  Run (ACLiC compilation recommended):
//      root -l -b -q 'Analysis.C+("TrueData")'
//
//  Output:
//      Plots/TestingControl/Histograms<name>.root    - all histograms (directories)
//          kinematics/   - lepton pair properties
//          classifier/   - classifier response
//          clusters_all/ - topoclusters for all tracks (before classification)
//          efficiency/   - topocluster-finding efficiency vs pT, eta, (pT,eta)
//                          (num/den histograms + ratios from TH1::Divide, "B")
//          matching/     - control plots of track-cluster matching (dR, dEta vs dPhi)
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
constexpr double kMaxPairPt   = 0.2;           // GeV
constexpr double kMinPairMass = 2.0;           // GeV
constexpr double kMaxPairMass = 3.5;           // GeV
constexpr double kConeDR      = 0.8;           // topocluster matching cone
constexpr float  kProbCut     = 0.5f;          // classifier threshold
constexpr float  kProbMargin  = 0.15f;         // dead zone around the threshold
constexpr int    kNYBins      = 5;             // pair |y| bins, 0.5 wide
constexpr int    kPrintEvery  = 200000;

// Efficiency binning
constexpr int    kEffPtBins   = 40;            // 1D vs pT
constexpr double kEffPtMin    = 1.0;           // GeV
constexpr double kEffPtMax    = 1.8;           // GeV
constexpr int    kEffEtaBins  = 50;            // 1D vs eta
constexpr int    kEff2DPtBins = 50;            // 2D (pT, eta) - coarser, fewer entries per bin
}  // namespace cfg

enum Species { kElectron = 0, kMuon = 1, kUnknown = 2 };

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
      count   = ::Book(d, prefix + "count",   "Matched clusters", 7, -0.5, 6.5);
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

// Set of histograms for one lepton species
struct LeptonHists {
   TH1D *eta, *energy, *pt, *phi;
   TH1D *pixelHits, *trtHits, *sctHits, *pixeldEdX;
   TH1D *dimass;
   std::array<TH1D*, cfg::kNYBins> dimassY;
   ClusterHists clusters;

   void Book(TDirectory* d, const std::string& label)
   {
      eta       = ::Book(d, "eta",       label + " #eta",      50, -3, 3);
      energy    = ::Book(d, "energy",    label + " energy",    50, 0, 12);
      pt        = ::Book(d, "pt",        label + " p_{T}",     50, 0, 2.8);
      phi       = ::Book(d, "phi",       label + " #phi",      30, -TMath::Pi(), TMath::Pi());
      pixelHits = ::Book(d, "pixelHits", "PixelHits",          11, -0.5, 10.5);
      trtHits   = ::Book(d, "trtHits",   "TRTHits",            51, -1, 50);
      sctHits   = ::Book(d, "sctHits",   "SCTHits",            12, 2.5, 14.5);
      pixeldEdX = ::Book(d, "pixeldEdX", "PixeldEdX",          100, 0, 2);
      dimass    = ::Book(d, "dimass",    "Lepton pair mass",   50, 2.5, 3.5);
      for (int b = 0; b < cfg::kNYBins; ++b)
         dimassY[b] = ::Book(d, Form("dimass_y%d", b),
                             Form("Lepton pair mass: |y| %g - %g", b * 0.5, b * 0.5 + 0.5),
                             40, 2.5, 3.5);
      clusters.Book(d, "clu_");
   }

   void FillTrack(const TLorentzVector& p, float pix, float trt, float sct, float dedx)
   {
      energy->Fill(p.E());
      eta->Fill(p.Eta());
      pt->Fill(p.Pt());
      phi->Fill(p.Phi());
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
   float Predict(std::array<float, 9>& x)
   {
      auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      std::array<int64_t, 2> shape{1, 9};
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

    const std::string& name = "TrueData";
    const std::string& inputFile = "../Data/data23_2trk_moreTCvars.root";//"../Data/mc_jpsi_ee.root","../Data/mc_jpsi_mumu.root"
    gROOT->SetBatch(kTRUE);
    gErrorIgnoreLevel = kError;

   // ---------------- Output file ----------------
    TFile histFile(Form("Plots/TestingControl/Histograms%s.root", name.c_str()), "RECREATE");

   // ---------------- Histograms ----------------
    TDirectory* dKin  = histFile.mkdir("kinematics");
    TDirectory* dClf  = histFile.mkdir("classifier");
    TDirectory* dAll  = histFile.mkdir("clusters_all");
    TDirectory* dEff  = histFile.mkdir("efficiency");
    TDirectory* dEl   = histFile.mkdir("electron");
    TDirectory* dMu   = histFile.mkdir("muon");
    TDirectory* dMatch = histFile.mkdir("matching");

    TH1D* hPairPt    = Book(dKin, "pt",       "Transverse momentum of lepton pair", 50, 0, 0.2);
    TH1D* hPairY     = Book(dKin, "rapidity", "Rapidity of lepton pair",            50, -3, 3);
    TH1D* hPairMass  = Book(dKin, "dimass",   "Lepton pair mass",                   50, 2.5, 3.5);
    TH1D* hDeltaPhi  = Book(dKin, "deltaPhi", "#Delta#phi [deg]",                   40, 160, 180);

    TH1D* hResponse     = Book(dClf, "response",       "RF response to experimental data", 50, 0, 1);
    TH1D* hRespOneFound = Book(dClf, "responseOneFound", "Response (one track matched)",   100, 0, 1);
    TH1D* hRespAmbig    = Book(dClf, "responseAmbiguous", "Response (ambiguous pair)",     100, 0, 1);

    ClusterHists allClusters;
    allClusters.Book(dAll);

    EffHists eff;
    eff.Book(dEff);

    // Control histograms of the track-cluster matching (filled inside matchCluster)
    TH1D* hDR      = Book(dMatch, "dR",      "#DeltaR(track, cluster), all clusters passing the cut;#DeltaR;Clusters", 120, 0, 6);
    TH1D* hPt      = Book(dMatch, "dPt",      "Transverse momentum cluster;#p_{T};Clusters", 120, 0, 6);
    TH1D* hDRBest  = Book(dMatch, "dR_best", "#DeltaR(track, selected cluster);#DeltaR;Tracks", 100, 0, cfg::kConeDR);
    dMatch->cd();
    TH2D* hDEtaDPhi = new TH2D("dEta_dPhi", "Track - cluster;#Delta#eta;#Delta#phi;Clusters",
                               120, -cfg::kConeDR, cfg::kConeDR, 64, -cfg::kConeDR, cfg::kConeDR);

    std::array<LeptonHists, 2> lepHists;
    lepHists[kElectron].Book(dEl, "Electron");
    lepHists[kMuon].Book(dMu, "Muon");

   // ---------------- Classifier ----------------
    RFClassifier classifier("../ONNX/RFMuonElectron.onnx");

   // ---------------- Input ----------------
    TChain chain("G2TauTree");
    chain.Add(inputFile.c_str());
    TTreeReader reader(&chain);

    TTreeReaderArray<unsigned int> TrackNum   (reader, "track_n");
    TTreeReaderArray<float>        TrackEta   (reader, "track_eta");
    TTreeReaderArray<float>        TrackPhi   (reader, "track_phi");
    TTreeReaderArray<float>        TrackPt    (reader, "track_pt");
    TTreeReaderArray<int>          TrackCharge(reader, "track_charge");

    TTreeReaderArray<float> TrackPixeldEdX(reader, "track_PixeldEdX");
    TTreeReaderArray<float> TrackPixelHits(reader, "track_PixelHits");
    TTreeReaderArray<float> TrackSCTHits  (reader, "track_SCTHits");
    TTreeReaderArray<float> TrackTRTHits  (reader, "track_TRTHits");

    TTreeReaderArray<unsigned int> TopoNum    (reader, "topoclus_n");
    TTreeReaderArray<float>        TopoEta    (reader, "topo_cluster_eta");
    TTreeReaderArray<float>        TopoPhi    (reader, "topo_cluster_phi");
    TTreeReaderArray<float>        TopoPt     (reader, "topo_cluster_pt");
    TTreeReaderArray<float>        TopoLambda (reader, "topo_cluster_lambda");
    TTreeReaderArray<float>        TopoLambda2(reader, "topo_cluster_lambda2");
    TTreeReaderArray<float>        TopoR2     (reader, "topo_cluster_r2");
    TTreeReaderArray<float>        TopoEMProb (reader, "topo_cluster_EM_prob");
    TTreeReaderArray<bool>         TopoPass   (reader, "topo_cluster_pass_sig_cut");

   // Topocluster matching: the highest-pT cluster in a cone of dR < kConeDR.
    auto matchCluster = [&](const TLorentzVector& p, bool fillControl = false) {
        ClusterVars c;
        int best = -1;
        double bestDR = -1;
        for (int t = 0; t < int(TopoPt.GetSize()); ++t) {
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

      // --- Track selection: 2 tracks, |eta|<2.5, pT>1 GeV ---
        if (int(TrackNum[0]) != 2) continue;

        std::array<TLorentzVector, 2> trk;
        bool tracksOk = true;
        for (int i = 0; i < 2; ++i) {
            if (!(std::fabs(TrackEta[i]) < cfg::kMaxAbsEta && TrackPt[i] > cfg::kMinTrackPt)) { tracksOk = false; break; }
            trk[i].SetPtEtaPhiM(TrackPt[i], TrackEta[i], TrackPhi[i], cfg::kMuonMass);
        }
        if (!tracksOk || TrackCharge[0] == TrackCharge[1]) continue;

      // --- Pair selection: pT < 0.2 GeV, 2.0 < M < 3.5 GeV ---
        const TLorentzVector pair = trk[0] + trk[1];
        if (pair.Perp() >= cfg::kMaxPairPt || pair.M() >= cfg::kMaxPairMass || cfg::kMinPairMass >= pair.M()) continue;

        hPairPt->Fill(pair.Perp());
        hPairY->Fill(pair.Rapidity());
        hPairMass->Fill(pair.M());
        hDeltaPhi->Fill(std::fabs(TVector2::Phi_mpi_pi(trk[0].Phi() - trk[1].Phi())) * TMath::RadToDeg());

      // --- e/mu classification for each track ---
        std::array<std::optional<float>, 2> response;
        for (int i = 0; i < 2; ++i) {
            ++allParticles;
            const ClusterVars cl = matchCluster(trk[i], true);   // fill control histograms here only
            allClusters.Fill(cl);
            eff.Fill(trk[i], cl.found);   // topocluster-finding efficiency (single track)
            if (!cl.found) continue;

            std::array<float, 9> x = {TrackPixelHits[i], TrackTRTHits[i], TrackSCTHits[i], TrackPixeldEdX[i],
                                    cl.f, cl.emProb, cl.lambda, cl.lambda2, cl.r2};
            response[i] = classifier.Predict(x);
            hResponse->Fill(*response[i]);
        }

      // --- Decision: mean response from available tracks ---
        const int nFound = int(response[0].has_value()) + int(response[1].has_value());
        Species species  = kUnknown;

        if (nFound == 0) {
            zeroFound += 2;
        } else {
            const float mean = (response[0].value_or(0.f) + response[1].value_or(0.f)) / nFound;
            if (nFound == 1) { oneFound += 2; hRespOneFound->Fill(mean); }
            else             { twoFound += 2; }

            if      (mean < cfg::kProbCut - cfg::kProbMargin) species = kElectron;
            else if (mean > cfg::kProbCut + cfg::kProbMargin) species = kMuon;
            else if (nFound == 2) { ambiguous += 2; hRespAmbig->Fill(*response[0]); }
        }
        if (species == kUnknown) continue;

      // --- Fill histograms for the classified pair ---
        LeptonHists& H = lepHists[species];
        H.dimass->Fill(pair.M());
        if (const int yb = YBin(pair.Rapidity()); yb >= 0) H.dimassY[yb]->Fill(pair.M());

        for (int i = 0; i < 2; ++i) {
            H.FillTrack(trk[i], TrackPixelHits[i], TrackTRTHits[i], TrackSCTHits[i], TrackPixeldEdX[i]);

            const ClusterVars cl = matchCluster(trk[i]);
            H.clusters.Fill(cl);
            if (cl.found) ++leptonCount[species];
        }
    }

   // ---------------- Efficiencies (binomial errors) ----------------
    eff.Finalize(dEff);

   // ---------------- Summary ----------------
    const float denom = allParticles > 0 ? allParticles : 1.f;
    std::cout << "===========================\n"
                << "End of file\n"
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
                << "===========================" << std::endl;

   // ---------------- Save output ----------------
    histFile.Write();
    histFile.Close();
}