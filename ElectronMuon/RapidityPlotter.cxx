// ============================================================================
//  RapidityPlotter.C  -  cluster features and RF response vs pair rapidity
//
//  Run:
//      root -l -b -q 'RapidityPlotter.C+'
//
//  Pages:
//    1) one page per feature, all samples on ONE plot:
//         Data electrons / Data muons / MC J/psi->ee / MC J/psi->mumu / pions
//       y axis = mean value of the feature in bins of pair rapidity y
//       (TProfile -> points, error on the mean).
//       Vertical dashed lines at |y| = kVLines (both signs).
//    2) pair rapidity distribution (kinematics/rapidity, plain TH1D),
//       normalised to unit area:  Data / MC ee / MC mumu / pions
//       (same vertical lines).
//    3) (last page) pair rapidity for events with 0.30 < mean RF response < 0.70
//       (kinematics/rapidity_mid), normalised to unit area. The legend shows
//       N and the % of all events (kinematics/rapidity) surviving this selection.
//
//  Input files = output of TestingMacro.C (Histograms<name>.root):
//      rapidity/<prefix><feature>_prof_y   (profiles)
//      kinematics/rapidity                 (distribution)
//      kinematics/rapidity_mid             (distribution, 0.30 < score < 0.70)
//  NOTE: the files must be produced with the version of TestingMacro.C that
//  contains the rapidity/ section and the rapidity_mid histogram.
//
//  Output: <outDir>/RapidityComparison.pdf (multi-page, with bookmarks)
// ============================================================================

#include <TAxis.h>
#include <TCanvas.h>
#include <TError.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TPad.h>
#include <TProfile.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TH2D.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace rpl {

// ----------------------------------------------------------------------------
//  Configuration  (EDIT PATHS HERE)
// ----------------------------------------------------------------------------
const std::string kDir      = "Plots/TestingControl/";
const std::string kFileData = kDir + "HistogramsTrueData.root";
const std::string kFileMCee = kDir + "HistogramsMC_ee.root";
const std::string kFileMCmm = kDir + "HistogramsMC_mumu.root";
const std::string kFilePion = kDir + "Histograms_SS_TrueData.root";

const std::string kProfSuffix = "_prof_y";
const int         kMinEntries = 10;      // bins with fewer entries are not drawn

// Vertical lines at these rapidity values (drawn at +y and -y)
const std::vector<double> kVLines = {1., 1.5, 2., 2.5};

const Color_t kColData   = kBlack;
const Color_t kColDataMu = kBlack;
const Color_t kColEl   = kRed + 1;
const Color_t kColMu   = kAzure + 2;
const Color_t kColPion = kGreen + 2;

// ----------------------------------------------------------------------------
//  Structures
// ----------------------------------------------------------------------------
struct Feature {            // quantity plotted vs y
   const char* name;        // histogram name without prefix / suffix
   const char* yTitle;      // axis title (mean value)
};

struct Entry {              // one curve in the plot
   TFile*      file;
   std::string path;        // directory + prefix, e.g. "rapidity/electron_"
                            // (for the distribution: full histogram path)
   std::string label;
   Color_t     col;
   int         marker;
   std::string path2 = "";  // optional: second directory + prefix, summed with `path`
   std::string refPath = "";// optional (distributions): reference histogram with ALL events;
                            // if set, the legend shows the % of events surviving the selection
};

const std::vector<Feature> kFeatures = {
   {"f",            "<F = p_{T}^{cl} / p_{T}^{trk}>"},
   {"emProb",       "<Cluster EM probability>"},
   {"lambda",       "<Cluster #lambda> / 10^{3}"},
   {"lambda2",      "<Cluster #lambda^{2}> / 10^{3}"},
   {"radius",       "<Cluster radius^{2}> / 10^{3}"},
   {"response",     "<RF response>"},
};

// ----------------------------------------------------------------------------
//  Utilities
// ----------------------------------------------------------------------------
std::string Uniq()
{
   static int n = 0;
   return Form("rpl_%d", n++);
}

TProfile* Get(TFile* f, const std::string& path)
{
   auto* h = dynamic_cast<TProfile*>(f->Get(path.c_str()));
   if (!h) std::cerr << "Profile not found: " << f->GetName() << " : " << path << std::endl;
   return h;
}

TH1D* GetH(TFile* f, const std::string& path)
{
   auto* h = dynamic_cast<TH1D*>(f->Get(path.c_str()));
   if (!h) std::cerr << "Histogram not found: " << f->GetName() << " : " << path << std::endl;
   return h;
}

// Profile of the feature vs y.
//  sigma == false : TProfile <path><name>_prof_y  (error = error on the mean)
//  sigma == true  : TH2D <path>response_vs_y -> ProfileX with option "s"
//                   (error = spread / sigma of the distribution in each y bin)
TProfile* LoadProfile(const Entry& e, const std::string& name, bool sigma)
{
   if (!sigma) {
      TProfile* p = Get(e.file, e.path + name + kProfSuffix);
      if (!p) return nullptr;
      if (!e.path2.empty()) {
         TProfile* p2 = Get(e.file, e.path2 + name + kProfSuffix);
         if (!p2) return nullptr;
         auto* sum = static_cast<TProfile*>(p->Clone(Uniq().c_str()));
         sum->Add(p2);
         p = sum;
      }
      return p;
   }

   auto get2 = [&](const std::string& dir) -> TH2D* {
      const std::string path = dir + "response_vs_y";
      auto* h = dynamic_cast<TH2D*>(e.file->Get(path.c_str()));
      if (!h) std::cerr << "TH2D not found: " << e.file->GetName() << " : " << path << std::endl;
      return h;
   };

   TH2D* h2 = get2(e.path);
   if (!h2) return nullptr;
   auto* hc = static_cast<TH2D*>(h2->Clone(Uniq().c_str()));
   if (!e.path2.empty()) {
      TH2D* h2b = get2(e.path2);
      if (!h2b) return nullptr;
      hc->Add(h2b);
   }
   // "s": bin error = standard deviation of the y-distribution in each x bin
   return hc->ProfileX(Uniq().c_str(), 1, -1, "s");
}

void SetStyle()
{
   gROOT->SetBatch(kTRUE);
   gErrorIgnoreLevel = kWarning;
   TH1::AddDirectory(kFALSE);

   gStyle->SetOptStat(0);
   gStyle->SetOptTitle(0);
   gStyle->SetEndErrorSize(0);

   gStyle->SetTextFont(42);
   gStyle->SetLabelFont(42, "XYZ");
   gStyle->SetTitleFont(42, "XYZ");
   gStyle->SetLabelSize(0.045, "XYZ");
   gStyle->SetTitleSize(0.05, "XYZ");
   gStyle->SetTitleOffset(1.25, "X");
   gStyle->SetTitleOffset(1.4, "Y");

   gStyle->SetPadLeftMargin(0.14);
   gStyle->SetPadRightMargin(0.05);
   gStyle->SetPadBottomMargin(0.13);
   gStyle->SetPadTopMargin(0.06);
   gStyle->SetPadTickX(1);
   gStyle->SetPadTickY(1);
   gStyle->SetLegendBorderSize(0);
   gStyle->SetLegendFillStyle(0);
   gStyle->SetLegendFont(42);
}

std::vector<std::pair<TCanvas*, std::string>>& Pages()
{
   static std::vector<std::pair<TCanvas*, std::string>> pages;
   return pages;
}

void WritePDF(const std::string& path)
{
   auto& p = Pages();
   if (p.empty()) { std::cerr << "No plots to save" << std::endl; return; }
   const size_t n = p.size();
   for (size_t i = 0; i < n; ++i) {
      const char* suffix = (n == 1) ? "" : (i == 0 ? "(" : (i == n - 1 ? ")" : ""));
      p[i].first->Print((path + suffix).c_str(), ("Title:" + p[i].second).c_str());
      delete p[i].first;
   }
   p.clear();
}

void Note(const std::string& s, double x, double y, double size = 0.042)
{
   TLatex t;
   t.SetNDC();
   t.SetTextFont(42);
   t.SetTextSize(size);
   t.DrawLatex(x, y, s.c_str());
}

// Vertical dashed lines at +-kVLines (only those inside the x range)
void DrawLines(double xlo, double xhi, double ymin, double ymax)
{
   for (double y : kVLines) {
      for (double sign : {+1., -1.}) {
         const double x = sign * y;
         if (x <= xlo || x >= xhi) continue;
         auto* l = new TLine(x, ymin, x, ymax);
         l->SetLineStyle(2);
         l->SetLineColor(kGray + 2);
         l->SetLineWidth(1);
         l->Draw();
      }
   }
}

// ----------------------------------------------------------------------------
//  One plot: all entries overlaid on a single canvas
// ----------------------------------------------------------------------------
void Compare(const Feature& v, const std::vector<Entry>& entries, bool sigma = false)
{
   // ---------------- Load profiles ----------------
   std::vector<TProfile*> ps;
   for (const auto& e : entries) {
      const std::string path = e.path + v.name;
      TProfile* p = LoadProfile(e, v.name, sigma);
      if (!p) return;
      if (p->GetEntries() <= 0) {
         std::cerr << "Empty profile: " << path << " (" << e.label << ")" << std::endl;
         return;
      }
      if (!ps.empty() && (p->GetNbinsX() != ps[0]->GetNbinsX() ||
                          p->GetXaxis()->GetXmin() != ps[0]->GetXaxis()->GetXmin() ||
                          p->GetXaxis()->GetXmax() != ps[0]->GetXaxis()->GetXmax())) {
         std::cerr << "Binning mismatch: " << path << std::endl;
         return;
      }
      ps.push_back(p);
   }

   const size_t N   = ps.size();
   const int    nb  = ps[0]->GetNbinsX();
   const double xlo = ps[0]->GetXaxis()->GetXmin();
   const double xhi = ps[0]->GetXaxis()->GetXmax();
   const double bw  = ps[0]->GetBinWidth(1);

   // ---------------- Graphs (small x offset so that the points do not overlap) ----------------
   std::vector<TGraphErrors*> g(N);
   double ylo = 1e30, yhi = -1e30;

   for (size_t i = 0; i < N; ++i) {
      const double dx = (double(i) - 0.5 * double(N - 1)) * 0.10 * bw;
      g[i] = new TGraphErrors();

      for (int b = 1; b <= nb; ++b) {
         if (ps[i]->GetBinEntries(b) < kMinEntries) continue;
         const double yy = ps[i]->GetBinContent(b);
         const double ee = ps[i]->GetBinError(b);
         g[i]->SetPoint(g[i]->GetN(), ps[i]->GetBinCenter(b) + dx, yy);
         g[i]->SetPointError(g[i]->GetN() - 1, 0, ee);
         ylo = std::min(ylo, yy - ee);
         yhi = std::max(yhi, yy + ee);
      }

      g[i]->SetLineColor(entries[i].col);
      g[i]->SetMarkerColor(entries[i].col);
      g[i]->SetMarkerStyle(entries[i].marker);
      g[i]->SetMarkerSize(1.0);
      g[i]->SetLineWidth(2);
   }

   if (ylo > yhi) { std::cerr << "No populated bins: " << v.name << std::endl; return; }

   // ---------------- Range ----------------
   double span = yhi - ylo;
   if (span <= 0) span = std::max(std::fabs(yhi), 1e-3);
   const double ymin = ylo - 0.15 * span;
   const double ymax = yhi + 0.35 * span;       // room for the legend

   // ---------------- Canvas ----------------
   auto* c = new TCanvas(Uniq().c_str(), "", 800, 800);

   auto* frame = new TH1D(Uniq().c_str(), "", nb, xlo, xhi);
   frame->SetMinimum(ymin);
   frame->SetMaximum(ymax);
   frame->GetXaxis()->SetTitle("y_{pair}");
   frame->GetYaxis()->SetTitle(v.yTitle);
   frame->Draw("AXIS");

   DrawLines(xlo, xhi, ymin, ymax);

   for (size_t i = N; i-- > 1;) g[i]->Draw("PE SAME");
   g[0]->Draw("PE SAME");                         // reference on top

   auto* leg = new TLegend(0.17, 0.77, 0.60, 0.93);
   leg->SetNColumns(1);
   leg->SetTextSize(0.018);
   for (size_t i = 0; i < N; ++i)
      leg->AddEntry(g[i], Form("%s (N = %.0f)", entries[i].label.c_str(), ps[i]->GetEntries()), "lep");
   leg->Draw();

   Pages().emplace_back(c, sigma ? std::string(v.name) + "_sigma" : std::string(v.name));
}

// ----------------------------------------------------------------------------
//  Pair rapidity distribution (plain TH1D), normalised to unit area.
//  entries[0] = Data (points), the rest = filled histograms.
//  If Entry::refPath is set, the legend also shows the % of events in the
//  reference histogram (all events) that survive the selection.
// ----------------------------------------------------------------------------
void CompareDist(const std::vector<Entry>& entries,
                 const std::string& page = "rapidity_distribution")
{
   std::vector<TH1D*>  hs;
   std::vector<double> nEntries;
   std::vector<double> nTotal;     // entries of the reference histogram (0 = none)

   for (const auto& e : entries) {
      TH1D* h0 = GetH(e.file, e.path);
      if (!h0) return;
      if (h0->Integral() <= 0) {
         std::cerr << "Empty histogram: " << e.path << " (" << e.label << ")" << std::endl;
         return;
      }
      if (!hs.empty() && (h0->GetNbinsX() != hs[0]->GetNbinsX() ||
                          h0->GetXaxis()->GetXmin() != hs[0]->GetXaxis()->GetXmin() ||
                          h0->GetXaxis()->GetXmax() != hs[0]->GetXaxis()->GetXmax())) {
         std::cerr << "Binning mismatch: " << e.path << " (" << e.label << ")" << std::endl;
         return;
      }

      double tot = 0;
      if (!e.refPath.empty()) {
         TH1D* hr = GetH(e.file, e.refPath);
         if (!hr) return;
         tot = hr->GetEntries();
      }

      nEntries.push_back(h0->GetEntries());
      nTotal.push_back(tot);
      auto* h = static_cast<TH1D*>(h0->Clone(Uniq().c_str()));
      h->Sumw2();
      hs.push_back(h);
   }

   // Normalise to unit area
   for (auto* h : hs) h->Scale(1. / h->Integral());

   // Style
   double ymaxAll = 0;
   for (size_t i = 0; i < hs.size(); ++i) {
      hs[i]->SetTitle("");
      hs[i]->SetLineColor(entries[i].col);
      hs[i]->SetMarkerColor(entries[i].col);
      hs[i]->SetMarkerStyle(entries[i].marker);
      hs[i]->SetMarkerSize(0.9);
      hs[i]->SetLineWidth(2);
      if (i == 0) hs[i]->SetFillStyle(0);
      else        hs[i]->SetFillColorAlpha(entries[i].col, 0.12);
      hs[i]->GetXaxis()->SetTitle("y_{pair}");
      hs[i]->GetYaxis()->SetTitle("Normalised entries");
      ymaxAll = std::max(ymaxAll, hs[i]->GetMaximum());
   }
   const double ymin = 0, ymax = ymaxAll * 1.4;
   const double xlo = hs[0]->GetXaxis()->GetXmin();
   const double xhi = hs[0]->GetXaxis()->GetXmax();

   auto* c = new TCanvas(Uniq().c_str(), "", 800, 800);

   hs[0]->SetMinimum(ymin);
   hs[0]->SetMaximum(ymax);
   hs[0]->Draw("AXIS");
   DrawLines(xlo, xhi, ymin, ymax);
   for (size_t i = 1; i < hs.size(); ++i) {
      hs[i]->Draw("HIST SAME");
      hs[i]->Draw("E1 SAME");
   }
   hs[0]->Draw("E1 SAME");                        // Data on top

   auto* leg = new TLegend(0.17, 0.77, 0.65, 0.93);
   leg->SetNColumns(1);
   leg->SetTextSize(0.018);
   for (size_t i = 0; i < hs.size(); ++i) {
      std::string txt;
      if (nTotal[i] > 0)
         txt = Form("%s (N = %.0f, %.1f%% of all events)", entries[i].label.c_str(),
                    nEntries[i], 100. * nEntries[i] / nTotal[i]);
      else
         txt = Form("%s (N = %.0f)", entries[i].label.c_str(), nEntries[i]);
      leg->AddEntry(hs[i], txt.c_str(), i == 0 ? "lep" : "lf");
   }
   leg->Draw();

   Pages().emplace_back(c, page);
}

}  // namespace rpl

// ============================================================================
//  Main function
// ============================================================================
void RapidityPlotter()
{
   using namespace rpl;
   SetStyle();

   TFile* fData = TFile::Open(kFileData.c_str(), "READ");
   TFile* fMCee = TFile::Open(kFileMCee.c_str(), "READ");
   TFile* fMCmm = TFile::Open(kFileMCmm.c_str(), "READ");
   TFile* fPion = TFile::Open(kFilePion.c_str(), "READ");
   for (TFile* f : {fData, fMCee, fMCmm, fPion}) {
      if (!f || f->IsZombie()) {
         std::cerr << "Cannot open one of the input files (check paths at the top of the macro)" << std::endl;
         return;
      }
   }

   // Pair rapidity distribution (kinematics/rapidity), normalised.
   // One histogram per file (before e/mu classification), so Data is a single curve.
   const std::vector<Entry> distEntries = {
      {fData, "kinematics/rapidity", "Data",                         kColData, 20},
      {fMCee, "kinematics/rapidity", "MC J/#psi #rightarrow ee",     kColEl,   21},
      {fMCmm, "kinematics/rapidity", "MC J/#psi #rightarrow #mu#mu", kColMu,   22},
      {fPion, "kinematics/rapidity", "Pions",                        kColPion, 23},
   };
   CompareDist(distEntries);

   // Curves on every feature plot. "all_" = all tracks (before e/mu classification),
   // "electron_" / "muon_" = tracks classified as e / mu.
   // Data = after selection, electrons and muons as separate curves.
   // (To sum them into one curve use path2: {..., "rapidity/electron_", ..., 20, "rapidity/muon_"}.)
   const std::vector<Entry> entries = {
      {fData, "rapidity/electron_", "Data electrons",             kColData,   25},
      {fData, "rapidity/muon_",     "Data muons",                 kColDataMu, 24},
      {fMCee, "rapidity/electron_", "MC J/#psi #rightarrow ee",   kColEl,   21},
      {fMCmm, "rapidity/muon_",     "MC J/#psi #rightarrow #mu#mu", kColMu, 20},
      {fPion, "rapidity/all_",      "Pions",                      kColPion, 23},
   };

   for (const auto& v : kFeatures) Compare(v, entries);

   // RF response with error bars = sigma of the distribution (from response_vs_y)
   const Feature respSigma = {"response", "<RF response> (error bars: #sigma)"};
   Compare(respSigma, entries, true);

   // Last page: pair rapidity for events with 0.30 < mean RF response < 0.70.
   // Legend: N and % of all events (kinematics/rapidity of the same file).
   const std::string all = "kinematics/rapidity";
   const std::vector<Entry> distMidEntries = {
      {fData, "kinematics/rapidity_mid", "Data",                         kColData, 20, "", all},
      {fMCee, "kinematics/rapidity_mid", "MC J/#psi #rightarrow ee",     kColEl,   21, "", all},
      {fMCmm, "kinematics/rapidity_mid", "MC J/#psi #rightarrow #mu#mu", kColMu,   22, "", all},
      {fPion, "kinematics/rapidity_mid", "Pions",                        kColPion, 23, "", all},
   };
   CompareDist(distMidEntries, "rapidity_mid_0.30_0.70");

   gSystem->mkdir(kDir.c_str(), kTRUE);
   WritePDF(kDir + "RapidityComparison.pdf");

   for (TFile* f : {fData, fMCee, fMCmm, fPion}) f->Close();
}