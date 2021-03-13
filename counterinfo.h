#ifndef COUNTERINFO_H
#define COUNTERINFO_H
#include <vector>

struct PixelContourDistStr{
  int num;
  int curLabel;
  std::vector<int>  adjLabel;
  std::vector<float> adjContourDist;

  PixelContourDistStr()
  {
      num = 0;
      curLabel = -1;
      adjLabel.clear();
      adjContourDist.clear();
  }
};

class PixelInfoMatrix
{
private:
    std::vector<PixelContourDistStr> _mat; // your matrix of CellXY
    int _r; // rows
    int _c; // cols

public:
    PixelInfoMatrix(int rows, int cols)
        : _r(rows), _c(cols)
    {
        _mat.clear();
        _mat.resize(_r*_c);
    }

    void set(int row, int col, const PixelContourDistStr& cell)
    {
        _mat[row * _c + col].num = cell.num;
        _mat[row * _c + col].curLabel = cell.curLabel;
        _mat[row * _c + col].adjLabel.assign(cell.adjLabel.begin(), cell.adjLabel.end());
        _mat[row * _c + col].adjContourDist.assign(cell.adjContourDist.begin(), cell.adjContourDist.end());;

    }

    PixelContourDistStr& get(int row, int col)
    {
        return _mat[row * _c + col];
    }
};

#endif // COUNTERINFO_H
